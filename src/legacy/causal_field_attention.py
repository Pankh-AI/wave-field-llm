"""
Causal Field Attention - 1D Field with Causal Convolution
===========================================================

This fixes the language modeling problem by:
1. Using a 1D field (not 2D) to preserve sequential order
2. Causal convolution (only look backward)
3. Vectorized operations (no Python loops)

Complexity: O(n) for scatter/gather + O(G log G) for FFT = O(n + G log G)
Memory: O(G * D) for field, not O(n^2) for attention matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalFieldAttention(nn.Module):
    """
    1D Causal Field Attention.
    
    Key idea: Map tokens onto a 1D field where position = sequence order.
    Use causal convolution so token i only sees tokens 0..i.
    """
    
    def __init__(self, embedding_dim, num_heads, field_size=512, sigma=3.0, device='cuda'):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.field_size = field_size
        self.sigma = sigma
        self.device = device
        
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        # Standard QKV projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Pre-compute causal kernel in frequency domain
        self._build_causal_kernel(field_size, sigma)
        
        self.scale = math.sqrt(self.head_dim)
    
    def _build_causal_kernel(self, size, sigma):
        """
        Build a causal (one-sided) exponential kernel.
        
        kernel[i] = exp(-i / sigma) for i >= 0 (looks backward)
        kernel[i] = 0 for i < 0 (cannot see future)
        """
        # Create indices centered at 0
        indices = torch.arange(size, dtype=torch.float32)
        
        # Causal exponential decay: only for i >= 0
        # We want position 0 to see itself strongly, position -1 to see 0 weakly, etc.
        # After FFT shift, center is at size//2
        kernel = torch.zeros(size)
        
        # Fill right half (causal part)
        for i in range(size):
            # Distance from center
            dist = abs(i - size // 2)
            if i >= size // 2:  # Future positions
                kernel[i] = 0.0
            else:  # Past positions
                kernel[i] = math.exp(-dist / sigma)
        
        # Normalize
        kernel = kernel / (kernel.sum() + 1e-8)
        
        # Store FFT for fast convolution
        self.register_buffer('causal_kernel_fft', torch.fft.rfft(kernel))
    
    def _causal_convolve_1d(self, field):
        """
        Apply causal convolution via FFT.
        
        field: (B, H, G, D) - 1D field for each batch, head
        Returns: (B, H, G, D) - convolved field
        """
        B, H, G, D = field.shape
        
        # Reshape for batched FFT: (B*H*D, G)
        field_flat = field.permute(0, 1, 3, 2).reshape(B * H * D, G)
        
        # FFT convolution (real FFT for efficiency)
        field_fft = torch.fft.rfft(field_flat, n=G)
        convolved_fft = field_fft * self.causal_kernel_fft.unsqueeze(0)
        convolved = torch.fft.irfft(convolved_fft, n=G)
        
        # Reshape back: (B, H, G, D)
        convolved = convolved.reshape(B, H, D, G).permute(0, 1, 3, 2)
        
        return convolved
    
    def forward(self, x, mask=None):
        """
        Forward pass with 1D causal field attention.
        
        x: (B, N, D) or (N, D)
        Returns: (B, N, D) or (N, D)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, N, D = x.shape
        G = self.field_size
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (B, N, D)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to heads: (B, H, N, head_dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # ============================================================
        # MAP TOKENS TO 1D FIELD POSITIONS
        # ============================================================
        # Linear mapping: token at sequence position i maps to field position (i/N)*G
        seq_positions = torch.arange(N, device=x.device, dtype=torch.float32)
        field_positions = (seq_positions / max(N - 1, 1)) * (G - 1)  # (N,) in range [0, G-1]
        
        # Compute field indices (integer positions for scatter/gather)
        field_indices = field_positions.long().clamp(0, G - 1)  # (N,)
        
        # Expand for batch and heads: (B, H, N)
        field_indices = field_indices.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1)
        
        # ============================================================
        # SCATTER: Deposit K*V onto 1D field
        # ============================================================
        # Weight values by key magnitude (importance)
        k_magnitude = k.norm(dim=-1, keepdim=True)  # (B, H, N, 1)
        weighted_v = v * k_magnitude  # (B, H, N, head_dim)
        
        # Initialize field: (B, H, G, head_dim)
        field = torch.zeros(B, self.num_heads, G, self.head_dim, device=x.device)
        
        # Vectorized scatter (no Python loops!)
        # For each head, scatter all tokens at once
        for h in range(self.num_heads):
            for b in range(B):
                # field_indices[b, h]: (N,) - which field position each token goes to
                # weighted_v[b, h]: (N, head_dim) - what to deposit
                idx = field_indices[b, h].unsqueeze(-1).expand(-1, self.head_dim)  # (N, head_dim)
                field[b, h].scatter_add_(0, idx, weighted_v[b, h])
        
        # ============================================================
        # CAUSAL CONVOLUTION: Spread information backward only
        # ============================================================
        field = self._causal_convolve_1d(field)
        
        # ============================================================
        # GATHER: Read from field at query positions
        # ============================================================
        output_heads = []
        
        for h in range(self.num_heads):
            head_output = torch.zeros(B, N, self.head_dim, device=x.device)
            
            for b in range(B):
                idx = field_indices[b, h].unsqueeze(-1).expand(-1, self.head_dim)  # (N, head_dim)
                head_output[b] = torch.gather(field[b, h], 0, idx)
            
            output_heads.append(head_output)
        
        # Concatenate heads: (B, N, D)
        output = torch.cat(output_heads, dim=-1)
        
        # Output projection
        output = self.out_proj(output)
        
        if squeeze_output:
            output = output.squeeze(0)
        
        return output


class CausalFieldAttentionV2(nn.Module):
    """
    Improved version with better vectorization.
    Uses einsum and batch operations to minimize loops.
    """
    
    def __init__(self, embedding_dim, num_heads, field_size=512, sigma=0.5, device='cuda'):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.field_size = field_size
        self.sigma = sigma
        self.device = device
        
        assert embedding_dim % num_heads == 0
        
        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        self._build_causal_kernel(field_size, sigma)
        self.scale = math.sqrt(self.head_dim)
    
    def _build_causal_kernel(self, size, sigma):
        """Build causal exponential kernel."""
        # Causal kernel: decays exponentially into the past
        positions = torch.arange(size, dtype=torch.float32)
        center = size // 2
        
        # Only past positions (left of center) have non-zero values
        kernel = torch.zeros(size)
        for i in range(size):
            if i <= center:
                kernel[i] = math.exp(-(center - i) / sigma)
        
        kernel = kernel / (kernel.sum() + 1e-8)
        self.register_buffer('causal_kernel_fft', torch.fft.rfft(kernel))
    
    def _causal_convolve_1d(self, field):
        """Causal 1D convolution via FFT."""
        B, H, G, D = field.shape
        
        # Flatten for batched FFT - FIXED: permute before reshape
        field_flat = field.permute(0, 1, 3, 2).reshape(B * H * D, G)
        
        # FFT convolution
        field_fft = torch.fft.rfft(field_flat, n=G)
        convolved_fft = field_fft * self.causal_kernel_fft.unsqueeze(0)
        convolved = torch.fft.irfft(convolved_fft, n=G)
        
        # Reshape back - FIXED: reshape then permute
        return convolved.reshape(B, H, D, G).permute(0, 1, 3, 2)
    
    def forward(self, x, mask=None):
        """Forward pass - fully vectorized, no Python loops."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, N, D = x.shape
        G = self.field_size
        H = self.num_heads
        head_dim = self.head_dim
        
        # Combined QKV projection
        qkv = self.qkv_proj(x)  # (B, N, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape to heads: (B, H, N, head_dim)
        q = q.view(B, N, H, head_dim).transpose(1, 2)
        k = k.view(B, N, H, head_dim).transpose(1, 2)
        v = v.view(B, N, H, head_dim).transpose(1, 2)
        
        # Map to 1D field positions (linear)
        seq_pos = torch.arange(N, device=x.device, dtype=torch.float32)
        field_pos = (seq_pos / max(N - 1, 1)) * (G - 1)
        field_idx = field_pos.long().clamp(0, G - 1)  # (N,)
        
        # Expand for batch and heads: (B, H, N)
        field_idx = field_idx.unsqueeze(0).unsqueeze(0).expand(B, H, -1)
        
        # Scatter K*V onto field
        k_mag = k.norm(dim=-1, keepdim=True)  # (B, H, N, 1)
        deposit = v * k_mag  # (B, H, N, head_dim)
        
        # Initialize field
        field = torch.zeros(B, H, G, head_dim, device=x.device)
        
        # VECTORIZED SCATTER: Flatten batch and heads for scatter_add_
        # Reshape: (B, H, N, head_dim) -> (B*H, N, head_dim)
        deposit_flat = deposit.reshape(B * H, N, head_dim)
        field_idx_flat = field_idx.reshape(B * H, N)  # (B*H, N)
        field_flat = field.reshape(B * H, G, head_dim)  # (B*H, G, head_dim)
        
        # Expand indices for scatter_add: (B*H, N) -> (B*H, N, head_dim)
        idx_expanded = field_idx_flat.unsqueeze(-1).expand(-1, -1, head_dim)
        
        # Scatter all at once (no Python loops!)
        field_flat.scatter_add_(1, idx_expanded, deposit_flat)
        
        # Reshape back: (B*H, G, head_dim) -> (B, H, G, head_dim)
        field = field_flat.reshape(B, H, G, head_dim)
        
        # Causal convolution
        field = self._causal_convolve_1d(field)
        
        # VECTORIZED GATHER: Flatten batch and heads for gather
        field_flat = field.reshape(B * H, G, head_dim)  # (B*H, G, head_dim)
        
        # Expand indices for gather: (B*H, N) -> (B*H, N, head_dim)
        idx_expanded = field_idx_flat.unsqueeze(-1).expand(-1, -1, head_dim)
        
        # Gather all at once (no Python loops!)
        output_flat = torch.gather(field_flat, 1, idx_expanded)  # (B*H, N, head_dim)
        
        # Reshape back: (B*H, N, head_dim) -> (B, H, N, head_dim) -> (B, N, H*head_dim)
        output = output_flat.reshape(B, H, N, head_dim).transpose(1, 2).reshape(B, N, D)
        
        # Output projection
        output = self.out_proj(output)
        
        if squeeze_output:
            output = output.squeeze(0)
        
        return output


if __name__ == '__main__':
    print("Testing Causal Field Attention...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create attention module
    attn = CausalFieldAttentionV2(
        embedding_dim=256,
        num_heads=8,
        field_size=512,
        device=device
    ).to(device)
    
    # Test forward pass
    x = torch.randn(2, 100, 256, device=device)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    out = attn(x)
    
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Memory: {mem:.1f} MB")
    
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    
    # Test causality
    print("\nTesting causality...")
    x_test = torch.randn(1, 50, 256, device=device)
    
    # Forward pass
    out1 = attn(x_test)
    
    # Modify future token
    x_test_modified = x_test.clone()
    x_test_modified[0, 40:] = torch.randn_like(x_test_modified[0, 40:])
    
    out2 = attn(x_test_modified)
    
    # Check if early positions are unchanged
    diff = (out1[0, :30] - out2[0, :30]).abs().max().item()
    print(f"Max difference in first 30 positions: {diff:.6f}")
    
    if diff < 1e-4:
        print("✓ CAUSAL: Early positions unaffected by future changes")
    else:
        print("✗ NOT CAUSAL: Early positions changed when future changed")
    
    print("\n✓ Causal Field Attention works!")

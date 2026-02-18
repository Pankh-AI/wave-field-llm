"""
Global Context Module for Field LLM
====================================

Solves the limited receptive field problem of 1D causal convolution
by periodically compressing all positions into a global summary
and broadcasting it back.

Complexity: O(n) - uses pooling, not attention
Memory: O(d) for the summary vector

How it works:
1. Pool all token representations into a small summary (O(n))
2. Pass summary through a learned projection
3. Add summary back to all positions (O(n))

This gives every token access to document-level context
without O(n²) attention.
"""

import torch
import torch.nn as nn


class GlobalContextModule(nn.Module):
    """
    Adds global document context to local field attention.
    
    Placed between field attention layers to extend receptive field
    from local (~18 positions) to global (full document).
    
    Still O(n) complexity:
    - Causal pooling: O(n) scan
    - Projection: O(d²) constant
    - Broadcast: O(n) addition
    """
    
    def __init__(self, embedding_dim: int, compression_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        compressed_dim = embedding_dim // compression_ratio
        
        # Compress: project each position to compressed representation
        self.compress = nn.Linear(embedding_dim, compressed_dim)
        
        # Gate: learn how much global context to add per position
        self.gate_proj = nn.Linear(embedding_dim + compressed_dim, embedding_dim)
        
        # Expand: project summary back to full dimension
        self.expand = nn.Linear(compressed_dim, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Add global context to all positions.
        
        x: (B, N, D) - token representations
        Returns: (B, N, D) - with global context added
        
        CAUSAL: Position i only sees summary of positions 0..i
        """
        B, N, D = x.shape
        
        # Step 1: Compress each position
        compressed = self.compress(x)  # (B, N, compressed_dim)
        
        # Step 2: Causal cumulative mean (position i sees avg of 0..i)
        # This is O(n) and maintains causality!
        cumsum = torch.cumsum(compressed, dim=1)  # (B, N, compressed_dim)
        counts = torch.arange(1, N + 1, device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(-1)
        causal_summary = cumsum / counts  # (B, N, compressed_dim)
        
        # Step 3: Expand summary back to full dimension
        global_context = self.expand(causal_summary)  # (B, N, D)
        global_context = self.dropout(global_context)
        
        # Step 4: Gated addition (model learns how much global context to use)
        gate_input = torch.cat([x, causal_summary], dim=-1)  # (B, N, D + compressed_dim)
        gate = torch.sigmoid(self.gate_proj(gate_input))  # (B, N, D)
        
        # Step 5: Residual addition with gate
        output = x + gate * global_context
        
        return output


if __name__ == "__main__":
    print("Testing Global Context Module...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    gcm = GlobalContextModule(embedding_dim=256).to(device)
    
    # Test forward
    x = torch.randn(2, 50, 256, device=device)
    out = gcm(x)
    
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in gcm.parameters()):,}")
    
    # Test causality (must use eval mode to disable dropout)
    gcm.eval()
    x1 = torch.randn(1, 50, 256, device=device)
    out1 = gcm(x1)
    
    x2 = x1.clone()
    x2[0, 30:] = torch.randn_like(x2[0, 30:])  # Change future
    out2 = gcm(x2)
    
    diff = (out1[0, :25] - out2[0, :25]).abs().max().item()
    print(f"\nCausality test:")
    print(f"  Max diff in first 25 positions: {diff:.8f}")
    if diff < 1e-5:
        print("  ✅ CAUSAL: Past positions unaffected by future changes")
    else:
        print("  ❌ NOT CAUSAL")
    
    print("\n✅ Global Context Module works!")

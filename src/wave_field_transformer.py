"""
Wave Field Transformer - Physics-Based Language Model
=====================================================

Field LLM V3: A genuinely new architecture that treats language as a
physical field system, not just a sequence of tokens.

What's NEW (vs any existing architecture):
1. Wave-parameterized attention (damped oscillation kernels, not arbitrary)
2. Content-dependent gating (adaptive attention patterns)
3. Multi-field coupling (cross-head field interactions)
4. Energy conservation (information can't be created/destroyed)
5. Field interference (constructive/destructive signal combination)

What this is NOT:
- Not a Mamba clone (no state space recurrence)
- Not a Hyena clone (physics-parameterized kernels, not implicit NN kernels)
- Not a standard transformer (O(n log n), not O(n²))

Complexity: O(n log n) per layer — between O(n) and O(n²)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import math
import sys
import os

if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.wave_field_attention import WaveFieldAttention
else:
    from .wave_field_attention import WaveFieldAttention


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, embedding_dim, max_cache=8192):
        super().__init__()
        self.embedding_dim = embedding_dim
        pe = self._make_pe(max_cache, embedding_dim)
        self.register_buffer('pe_cache', pe)
    
    def _make_pe(self, length, dim):
        position = torch.arange(length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, seq_len, device):
        if seq_len <= self.pe_cache.shape[0]:
            return self.pe_cache[:seq_len].to(device)
        return self._make_pe(seq_len, self.embedding_dim).to(device)


class WaveFieldTransformerLayer(nn.Module):
    """
    Single layer of the Wave Field Transformer.
    
    Structure:
    1. Wave Field Attention (physics-based, content-dependent)
    2. Feed-Forward Network (standard)
    3. Pre-norm residual connections
    """
    
    def __init__(self, embedding_dim=256, num_heads=8, ffn_dim=1024,
                 field_size=512, max_seq_len=128, dropout=0.1,
                 n_components=1, local_window=0, device='cuda'):
        super().__init__()

        self.attention = WaveFieldAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            field_size=field_size,
            max_seq_len=max_seq_len,
            n_components=n_components,
            local_window=local_window,
            device=device
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embedding_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-norm attention with residual
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        # Pre-norm FFN with residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        return x


class FieldInterferenceModule(nn.Module):
    """
    Field Interference: Models constructive and destructive combination
    of signals from different layers.
    
    In physics, when two waves meet:
    - Same phase → constructive (amplify)
    - Opposite phase → destructive (cancel)
    
    This module lets the model learn which signals to amplify and which
    to cancel, providing a physics-based information routing mechanism.
    
    Replaces the simple GlobalContextModule with interference-based mixing.
    """
    
    def __init__(self, embedding_dim, dropout=0.1, initial_temperature=-2.0):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Phase detector: determines the "phase" of each position's signal
        self.local_phase_proj = nn.Linear(embedding_dim, embedding_dim)
        self.global_phase_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # V3.3: diverse temperature init — sharp vs soft for different modules
        self.interference_temperature = nn.Parameter(torch.tensor(initial_temperature))
        
        # Interference gate: controls constructive vs destructive
        self.interference_gate = nn.Linear(embedding_dim * 2, embedding_dim)
        
        # Global field summary (causal cumulative mean)
        self.compress = nn.Linear(embedding_dim, embedding_dim // 4)
        self.expand = nn.Linear(embedding_dim // 4, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x):
        """
        Apply field interference. V3.1: sharper, more selective.
        
        x: (B, N, D) — token representations
        Returns: (B, N, D) — with interference applied
        """
        B, N, D = x.shape
        
        # Compute causal global context (O(n) — cumulative mean)
        compressed = self.compress(x)
        cumsum = torch.cumsum(compressed, dim=1)
        counts = torch.arange(1, N + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        global_ctx = self.expand(cumsum / counts)
        global_ctx = self.dropout(global_ctx)
        
        # V3.1: Separate phase projections for local and global
        local_phase = F.normalize(self.local_phase_proj(x), dim=-1)
        global_phase = F.normalize(self.global_phase_proj(global_ctx), dim=-1)
        
        # Cosine similarity between local and global phases
        phase_alignment = (local_phase * global_phase).sum(dim=-1, keepdim=True)
        
        # V3.1: Temperature-scaled sigmoid for SHARPER interference
        # Low temperature → sharp (near binary: amplify or suppress)
        # V3.0 used sqrt(D) scaling which was too soft
        temp = F.softplus(self.interference_temperature) + 0.05
        interference_strength = torch.sigmoid(phase_alignment / temp)
        
        # Gate: combine local and global
        gate_input = torch.cat([x, global_ctx], dim=-1)
        gate = torch.sigmoid(self.interference_gate(gate_input))
        
        # Apply interference: amplify aligned, suppress misaligned
        output = x + gate * global_ctx * interference_strength
        
        return output


class WaveFieldTransformer(nn.Module):
    """
    Wave Field Transformer for Language Modeling.
    
    A physics-based language model where:
    - Tokens are mapped to a continuous field
    - Information propagates via wave dynamics (not convolution)
    - Different heads = different fields with different physics
    - Fields interact through coupling
    - Energy is conserved (anti-hallucination)
    - Interference patterns route information
    
    Drop-in replacement for CausalFieldTransformer — same interface.
    """
    
    def __init__(self,
                 vocab_size=50257,
                 embedding_dim=256,
                 num_layers=6,
                 num_heads=8,
                 ffn_dim=1024,
                 field_size=512,
                 max_seq_len=2048,
                 dropout=0.1,
                 use_checkpoint=False,
                 interference_interval=3,
                 n_components=1,
                 local_window=0,
                 device=None):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.use_checkpoint = use_checkpoint
        self.interference_interval = interference_interval
        self.device = device if device is not None else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(
            embedding_dim, max_cache=max_seq_len
        )
        self.dropout = nn.Dropout(dropout)
        
        # Wave Field Transformer layers
        self.layers = nn.ModuleList([
            WaveFieldTransformerLayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                field_size=field_size,
                max_seq_len=max_seq_len,
                dropout=dropout,
                n_components=n_components,
                local_window=local_window,
                device=self.device
            )
            for _ in range(num_layers)
        ])
        
        # Field Interference modules (inserted periodically)
        # V3.3: diverse temperatures — sharp early (binary decisions),
        # softer later (nuanced refinement)
        num_interference = num_layers // interference_interval
        interference_temps = [-2.0, 0.5]
        self.interference_modules = nn.ModuleList([
            FieldInterferenceModule(
                embedding_dim=embedding_dim, dropout=dropout,
                initial_temperature=interference_temps[i % len(interference_temps)]
            )
            for i in range(num_interference)
        ])
        
        # Output
        self.norm = nn.LayerNorm(embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocab_size, bias=False)
        
        # Tie weights
        self.output_projection.weight = self.token_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # V4.3: Re-apply attention-specific initialization that the generic
        # init above overwrites.
        from .wave_field_attention import LearnedFeatureMap
        for layer in self.layers:
            attn = layer.attention
            D = self.embedding_dim

            with torch.no_grad():
                # Gate rows (last D) — bias=2.0, weight=0 (gates start open)
                attn.qkvg_proj.weight[3 * D:].zero_()
                attn.qkvg_proj.bias[3 * D:].fill_(2.0)

                # Learned feature maps: restore identity init (generic init
                # overwrites with normal(0, 0.02), destroying the identity)
                nn.init.eye_(attn.q_feature_map.linear.weight)
                nn.init.zeros_(attn.q_feature_map.linear.bias)
                nn.init.eye_(attn.k_feature_map.linear.weight)
                nn.init.zeros_(attn.k_feature_map.linear.bias)

                # Spectral gate: restore near-zero output init (generic init
                # sets std=0.02, we need ~0.001 so gate starts at ≈0)
                nn.init.normal_(attn.spectral_gate.net[-1].weight, 0, 0.001)
                nn.init.zeros_(attn.spectral_gate.net[-1].bias)
    
    def forward(self, input_ids, labels=None, mask=None):
        """
        Forward pass.
        
        input_ids: (B, N) — token indices
        labels: (B, N) — target token indices (for training)
        mask: (B, N) — attention mask (optional)
        
        Returns: logits (B, N, vocab_size) and loss (if labels provided)
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        B, N = input_ids.shape
        
        # Embeddings + positional encoding
        x = self.token_embedding(input_ids)
        pos_enc = self.positional_encoding(N, input_ids.device)
        x = x + pos_enc.unsqueeze(0)
        x = self.dropout(x)
        
        # Wave Field layers with interference
        interference_idx = 0
        for i, layer in enumerate(self.layers):
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, mask, use_reentrant=False
                )
            else:
                x = layer(x, mask)
            
            # Apply field interference periodically
            if ((i + 1) % self.interference_interval == 0 and
                    interference_idx < len(self.interference_modules)):
                x = self.interference_modules[interference_idx](x)
                interference_idx += 1
        
        # Output
        x = self.norm(x)
        logits = self.output_projection(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
        
        return logits, loss


if __name__ == '__main__':
    print("Testing Wave Field Transformer...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = WaveFieldTransformer(
        vocab_size=256,
        embedding_dim=256,
        num_layers=6,
        num_heads=8,
        ffn_dim=1024,
        field_size=512,
        device=device
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")
    
    # Test forward
    x = torch.randint(0, 256, (2, 100), device=device)
    y = torch.randint(0, 256, (2, 100), device=device)
    
    logits, loss = model(x, labels=y)
    
    print(f"Input:  {x.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Loss:   {loss.item():.3f}")
    print("Wave Field Transformer works!")

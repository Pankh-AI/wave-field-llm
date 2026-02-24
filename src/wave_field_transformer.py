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
                 n_components=1, local_window=0, use_analytic_kernel=True,
                 feature_map_depth=2, use_write_gate=True,
                 use_3d_interference=False,
                 use_kernel_mixture=False, num_basis_kernels=4,
                 layer_idx=0, num_layers=1,
                 device='cuda'):
        super().__init__()

        self.attention = WaveFieldAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            field_size=field_size,
            max_seq_len=max_seq_len,
            n_components=n_components,
            local_window=local_window,
            use_analytic_kernel=use_analytic_kernel,
            feature_map_depth=feature_map_depth,
            use_write_gate=use_write_gate,
            use_3d_interference=use_3d_interference,
            use_kernel_mixture=use_kernel_mixture,
            num_basis_kernels=num_basis_kernels,
            layer_idx=layer_idx,
            num_layers=num_layers,
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
                 device=None,
                 use_analytic_kernel=True,
                 feature_map_depth=2,
                 residual_scale=False,
                 use_write_gate=True,
                 use_3d_interference=False,
                 use_kernel_mixture=False,
                 num_basis_kernels=4):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.use_checkpoint = use_checkpoint
        self.interference_interval = interference_interval
        self.residual_scale = residual_scale
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
                use_analytic_kernel=use_analytic_kernel,
                feature_map_depth=feature_map_depth,
                use_write_gate=use_write_gate,
                use_3d_interference=use_3d_interference,
                use_kernel_mixture=use_kernel_mixture,
                num_basis_kernels=num_basis_kernels,
                layer_idx=layer_idx,
                num_layers=num_layers,
                device=self.device
            )
            for layer_idx in range(num_layers)
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

        # Re-apply attention-specific initialization that the generic
        # init above overwrites.
        H = self.num_heads
        D = self.embedding_dim
        head_dim = D // H

        for layer in self.layers:
            attn = layer.attention

            with torch.no_grad():
                # Gate rows (last D) — bias=2.0, weight=0 (gates start open)
                attn.qkvg_proj.weight[3 * D:].zero_()
                attn.qkvg_proj.bias[3 * D:].fill_(2.0)

                # Learned feature maps: restore identity init for all Linear layers
                # (generic init above overwrites with normal_(0, 0.02))
                for fm in [attn.q_feature_map, attn.k_feature_map]:
                    for module in fm.net:
                        if isinstance(module, nn.Linear):
                            nn.init.eye_(module.weight)
                            nn.init.zeros_(module.bias)

                # Spectral gate: restore near-zero output init (skip if kernel mixture)
                if attn.spectral_gate is not None:
                    nn.init.normal_(attn.spectral_gate.net[-1].weight, 0, 0.001)
                    nn.init.zeros_(attn.spectral_gate.net[-1].bias)

                # Kernel mixture: restore zero init for projection, warm bias for basis 1
                if hasattr(attn, 'kernel_mix_proj'):
                    attn.kernel_mix_proj.data.zero_()
                if hasattr(attn, 'kernel_mix_bias'):
                    K = attn.kernel_mix_bias.shape[-1]
                    attn.kernel_mix_bias.data.zero_()
                    if K > 1:
                        attn.kernel_mix_bias.data[:, 1] = 5.0  # 98% on standard HiPPO

                # V4.4: 3D interference — larger token position scale for
                # spatial diversity (generic init gives std=0.02, too small)
                if hasattr(attn, 'token_pos_proj'):
                    nn.init.normal_(attn.token_pos_proj.weight, 0, 0.1)
                    nn.init.zeros_(attn.token_pos_proj.bias)

        # V4.2-D: Residual scaling — scale down residual contribution at init
        # to stabilize deep networks (GPT-style 1/sqrt(2*num_layers)).
        if self.residual_scale:
            scale = (2 * len(self.layers)) ** -0.5
            for layer in self.layers:
                with torch.no_grad():
                    layer.attention.out_proj.weight.mul_(scale)
                    layer.ffn[3].weight.mul_(scale)  # second Linear in FFN

    def configure_optimizer(self, base_lr, weight_decay=0.01,
                            qk_lr_mult=3.0, kernel_lr_mult=50.0):
        """Create AdamW with per-group learning rates.

        Three param groups (V4.3.2):
        1. Other params: base_lr (default)
        2. QKV projections: base_lr × 3 (V4.2 ablation: -21% PPL)
        3. Kernel physics params: base_lr × 50, weight_decay=0
           (S4/Mamba practice — monitor showed 27-80x gradient deficit)
        """
        kernel_names = {'wave_frequency', 'wave_damping', 'wave_phase'}
        kernel_params = []
        qk_params = []
        other_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            param_leaf = name.split('.')[-1]
            if param_leaf in kernel_names:
                kernel_params.append(param)
            elif 'qkvg_proj' in name:
                qk_params.append(param)
            else:
                other_params.append(param)

        return torch.optim.AdamW([
            {'params': other_params, 'lr': base_lr, 'weight_decay': weight_decay},
            {'params': qk_params, 'lr': base_lr * qk_lr_mult, 'weight_decay': weight_decay},
            {'params': kernel_params, 'lr': base_lr * kernel_lr_mult, 'weight_decay': 0.0},
        ])

    def compile_model(self, mode='reduce-overhead'):
        """Compile non-FFT submodules with torch.compile for Inductor fusion.

        Call AFTER model.to(device) and _init_weights (both happen in __init__).
        Only compiles code paths that Inductor handles well — FFN, LayerNorm,
        and feature maps. The FFT path (complex tensors) is left uncompiled
        because Inductor lacks complex tensor codegen (PyTorch issue #125718).

        Args:
            mode: 'default' (safest), 'reduce-overhead' (CUDA graphs),
                  'max-autotune' (slow compile, fastest run)
        Returns: self (for chaining)
        """
        if not hasattr(torch, 'compile'):
            return self

        for layer in self.layers:
            layer.ffn = torch.compile(layer.ffn, mode=mode)
            layer.norm1 = torch.compile(layer.norm1, mode=mode)
            layer.norm2 = torch.compile(layer.norm2, mode=mode)
            # Feature maps: ELU+1 + Linear — fully compile-safe
            layer.attention.q_feature_map = torch.compile(
                layer.attention.q_feature_map, mode=mode
            )
            layer.attention.k_feature_map = torch.compile(
                layer.attention.k_feature_map, mode=mode
            )
        return self

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

    # Test kernel mixture mode
    print("\nTesting Kernel Mixture mode...")
    model_km = WaveFieldTransformer(
        vocab_size=256, embedding_dim=256, num_layers=6, num_heads=8,
        ffn_dim=1024, field_size=512,
        use_kernel_mixture=True, num_basis_kernels=4,
        use_write_gate=False, device=device
    ).to(device)
    km_params = sum(p.numel() for p in model_km.parameters())
    print(f"Params (kernel mixture): {km_params:,} (vs {param_count:,} SpectralGate)")
    logits_km, loss_km = model_km(x, labels=y)
    print(f"Loss:   {loss_km.item():.3f}")
    # Verify mixing weights are zero-init
    mix_max = model_km.layers[0].attention.kernel_mix_proj.abs().max().item()
    print(f"kernel_mix_proj max: {mix_max:.6f} (should be 0)")
    print("Kernel Mixture works!")

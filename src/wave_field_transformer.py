"""
Wave Field Transformer - Physics-Based Language Model
=====================================================

Hybrid architecture combining O(n log n) wave field attention with optional
standard O(n²) attention layers and O(n) GLA (gated linear attention) layers.

Core: tokens scatter onto a continuous 1D field, damped wave kernels convolve
via FFT, cross-head coupling mixes fields, results gather back at token
positions. Content-dependent gating controls output.

Supports:
- Pure wave field model (all layers wave-based)
- Hybrid: wave + standard attention at specified layer indices
- Hybrid: wave + GLA with Zamba2-style weight sharing
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

# GLA import for hybrid architecture
try:
    if __name__ == '__main__':
        from src.gla import GLALayer
    else:
        from .gla import GLALayer
except ImportError:
    GLALayer = None


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
    """Pre-norm wave field attention + FFN with residual connections."""

    def __init__(self, embedding_dim=256, num_heads=8, ffn_dim=1024,
                 field_size=512, max_seq_len=128, dropout=0.1,
                 n_components=1, use_analytic_kernel=True,
                 feature_map_depth=2,
                 layer_idx=0, num_layers=1,
                 skip_causal_enforce=False,
                 n_frozen_heads=0,
                 use_spectral_gate=None,
                 n_attn_heads=0,
                 device='cuda'):
        super().__init__()

        self.attention = WaveFieldAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            field_size=field_size,
            max_seq_len=max_seq_len,
            n_components=n_components,
            use_analytic_kernel=use_analytic_kernel,
            feature_map_depth=feature_map_depth,
            layer_idx=layer_idx,
            num_layers=num_layers,
            skip_causal_enforce=skip_causal_enforce,
            n_frozen_heads=n_frozen_heads,
            use_spectral_gate=use_spectral_gate,
            n_attn_heads=n_attn_heads,
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

    Physics-based language model: tokens scatter onto a continuous 1D field,
    information propagates via damped wave kernels (FFT convolution), and
    results gather back. O(n log n) per wave layer.

    Supports hybrid architectures: wave layers + standard attention layers
    (at positions specified by hybrid_attention_layers) + GLA layers
    (at positions specified by gla_layers, with Zamba2-style weight sharing).
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
                 device=None,
                 use_analytic_kernel=True,
                 feature_map_depth=2,
                 residual_scale=False,
                 skip_causal_enforce=False,
                 n_frozen_heads=0,
                 use_spectral_gate=None,
                 n_attn_heads=0,
                 hybrid_attention_layers=None,
                 gla_layers=None):
        super().__init__()
        self.n_attn_heads = n_attn_heads

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.use_checkpoint = use_checkpoint
        self.interference_interval = interference_interval
        self.residual_scale = residual_scale
        self._hybrid_indices = set(hybrid_attention_layers or [])
        self._gla_indices = set(gla_layers or [])
        self._gla_shared_source = {}  # track which GLA layers share weights
        self._hybrid_shared_source = {}  # track which hybrid layers share weights
        self.device = device if device is not None else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(
            embedding_dim, max_cache=max_seq_len
        )
        self.dropout = nn.Dropout(dropout)

        # V5.0: Hybrid layer construction — wave field for most layers,
        # standard attention for layers in hybrid_attention_layers,
        # GLA layers for layers in gla_layers (with Zamba2 weight sharing).
        self.layers = nn.ModuleList()
        self._gla_source_layer = None
        self._hybrid_source_layer = None
        for layer_idx in range(num_layers):
            if layer_idx in self._gla_indices:
                if GLALayer is None:
                    raise ImportError("GLALayer not available — check src/gla.py")
                if self._gla_source_layer is None:
                    gla = GLALayer(
                        d_model=embedding_dim,
                        num_heads=num_heads,
                        ffn_dim=ffn_dim,
                        dropout=dropout,
                    )
                    self._gla_source_layer = gla
                    self.layers.append(gla)
                else:
                    # Weight sharing: subsequent GLA layers reuse the first
                    self.layers.append(self._gla_source_layer)
                    self._gla_shared_source[layer_idx] = True
            elif layer_idx in self._hybrid_indices:
                if self._hybrid_source_layer is None:
                    attn_layer = nn.TransformerEncoderLayer(
                        d_model=embedding_dim, nhead=num_heads,
                        dim_feedforward=ffn_dim, dropout=dropout,
                        activation='gelu', batch_first=True, norm_first=True
                    )
                    self._hybrid_source_layer = attn_layer
                    self.layers.append(attn_layer)
                else:
                    # Zamba2-style weight sharing: reuse first hybrid layer's params
                    self.layers.append(self._hybrid_source_layer)
                    self._hybrid_shared_source[layer_idx] = True
            else:
                self.layers.append(WaveFieldTransformerLayer(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    field_size=field_size,
                    max_seq_len=max_seq_len,
                    dropout=dropout,
                    n_components=n_components,
                    use_analytic_kernel=use_analytic_kernel,
                    feature_map_depth=feature_map_depth,
                    layer_idx=layer_idx,
                    num_layers=num_layers,
                    skip_causal_enforce=skip_causal_enforce,
                    n_frozen_heads=n_frozen_heads,
                    use_spectral_gate=use_spectral_gate,
                    n_attn_heads=n_attn_heads,
                    device=self.device
                ))
        
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
        # init above overwrites. Skip standard attention layers (V5.0 hybrid).
        H = self.num_heads
        D = self.embedding_dim
        head_dim = D // H

        for i, layer in enumerate(self.layers):
            if i in self._hybrid_indices or i in self._gla_indices:
                continue  # nn.TransformerEncoderLayer / GLALayer handle their own init
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

                # V4.5.0: Cross-dim mixing — restore identity init
                nn.init.eye_(attn.cross_dim.weight)

                # Spectral gate: restore meaningful init
                if attn.spectral_gate is not None:
                    nn.init.normal_(attn.spectral_gate.net[-1].weight, 0, 0.02)
                    nn.init.zeros_(attn.spectral_gate.net[-1].bias)

        # V4.2-D: Residual scaling — scale down residual contribution at init
        # to stabilize deep networks (GPT-style 1/sqrt(2*num_layers)).
        if self.residual_scale:
            scale = (2 * len(self.layers)) ** -0.5
            for i, layer in enumerate(self.layers):
                if i in self._hybrid_indices or i in self._gla_indices:
                    continue  # standard / GLA layer has its own scaling
                with torch.no_grad():
                    layer.attention.out_proj.weight.mul_(scale)
                    layer.ffn[3].weight.mul_(scale)  # second Linear in FFN

    def configure_optimizer(self, base_lr, weight_decay=0.01,
                            qk_lr_mult=3.0, kernel_lr_mult=50.0):
        """Create AdamW with per-group learning rates.

        3 param groups:
        1. Other params: base_lr (default)
        2. QKV projections (wave qkvg_proj + GLA/hybrid Q/K/V): base_lr × qk_lr_mult
        3. Kernel physics params + SpectralGate: base_lr × kernel_lr_mult, wd=0
        """
        kernel_names = {'wave_frequency', 'wave_damping', 'wave_phase'}
        # Match QKV projections across all layer types
        qk_proj_names = {'qkvg_proj', 'q_proj', 'k_proj', 'v_proj',
                         'in_proj_weight', 'in_proj_bias'}
        kernel_params = []
        qk_params = []
        other_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            param_leaf = name.split('.')[-1]
            if param_leaf in kernel_names:
                kernel_params.append(param)
            elif 'spectral_gate' in name:
                kernel_params.append(param)
            elif any(qk_name in name for qk_name in qk_proj_names):
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

        # Compile GLA source layer first, then update all shared references
        compiled_gla = None
        for i, layer in enumerate(self.layers):
            if i in self._gla_indices:
                if i not in self._gla_shared_source:
                    # This is the source GLA layer — compile it
                    compiled_gla = torch.compile(layer, mode=mode)
                    self.layers[i] = compiled_gla
                else:
                    # Shared layer — point to the compiled source
                    self.layers[i] = compiled_gla
                continue
            if i in self._hybrid_indices:
                if i not in self._hybrid_shared_source:
                    compiled_hybrid = torch.compile(layer, mode=mode)
                    self.layers[i] = compiled_hybrid
                else:
                    self.layers[i] = compiled_hybrid
                continue
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
        
        # Wave Field layers with interference (V5.0: hybrid-aware, V4.7: GLA-aware)
        interference_idx = 0
        causal_mask = None  # lazily built for standard attention layers
        for i, layer in enumerate(self.layers):
            is_hybrid = i in self._hybrid_indices
            is_gla = i in self._gla_indices
            if is_hybrid:
                # Standard nn.TransformerEncoderLayer — needs causal mask
                if causal_mask is None or causal_mask.shape[0] != N:
                    causal_mask = nn.Transformer.generate_square_subsequent_mask(
                        N, device=x.device, dtype=x.dtype
                    )
                if self.use_checkpoint and self.training:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, causal_mask, None, True,
                        use_reentrant=False
                    )
                else:
                    x = layer(x, src_mask=causal_mask, is_causal=True)
            elif is_gla:
                # GLALayer — causal by construction, no mask needed
                if self.use_checkpoint and self.training:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, use_reentrant=False
                    )
                else:
                    x = layer(x)
            else:
                # WaveFieldTransformerLayer — causality enforced via FFT
                if self.use_checkpoint and self.training:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, mask, use_reentrant=False
                    )
                else:
                    x = layer(x, mask)

            # Apply field interference periodically (skip after hybrid/GLA layers —
            # they already have global context from O(n²) attention or recurrence)
            if ((i + 1) % self.interference_interval == 0 and
                    interference_idx < len(self.interference_modules)
                    and not is_hybrid and not is_gla):
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

    # Test hybrid mode
    print("\nTesting Hybrid mode (standard attn at layer 3)...")
    model_h = WaveFieldTransformer(
        vocab_size=256, embedding_dim=256, num_layers=6, num_heads=8,
        ffn_dim=1024, field_size=512,
        hybrid_attention_layers=[3], device=device
    ).to(device)
    h_params = sum(p.numel() for p in model_h.parameters())
    print(f"Params (hybrid): {h_params:,} (vs {param_count:,} pure wave)")
    logits_h, loss_h = model_h(x, labels=y)
    print(f"Loss:   {loss_h.item():.3f}")
    print("Hybrid mode works!")

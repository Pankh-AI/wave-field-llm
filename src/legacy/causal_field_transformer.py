"""
Causal Field Transformer - Full Language Model
================================================

Uses 1D causal field attention for proper sequential modeling.
Supports BPE tokenization for word-level language modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

# Handle both direct execution and import
if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.legacy.causal_field_attention import CausalFieldAttentionV2
    from src.global_context import GlobalContextModule
else:
    from .causal_field_attention import CausalFieldAttentionV2
    from ..global_context import GlobalContextModule


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding - works for any length."""
    
    def __init__(self, embedding_dim, max_cache=8192):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Pre-cache common lengths
        pe = self._make_pe(max_cache, embedding_dim)
        self.register_buffer('pe_cache', pe)
    
    def _make_pe(self, length, dim):
        """Generate sinusoidal positional encodings."""
        position = torch.arange(length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, seq_len, device):
        """Get positional encoding for given sequence length."""
        if seq_len <= self.pe_cache.shape[0]:
            return self.pe_cache[:seq_len].to(device)
        else:
            # Generate on the fly for very long sequences
            return self._make_pe(seq_len, self.embedding_dim).to(device)


class CausalFieldTransformerLayer(nn.Module):
    """Single transformer layer with causal field attention."""
    
    def __init__(self,
                 embedding_dim=256,
                 num_heads=8,
                 ffn_dim=1024,
                 field_size=512,
                 dropout=0.1,
                 device='cuda'):
        super().__init__()
        
        self.attention = CausalFieldAttentionV2(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            field_size=field_size,
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
        # Attention with residual
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        return x


class CausalFieldTransformer(nn.Module):
    """
    Causal Field Transformer for Language Modeling.
    
    Features:
    - 1D causal field attention (preserves sequential order)
    - Sinusoidal positional encoding (unlimited length)
    - BPE tokenizer support
    - Gradient checkpointing for memory efficiency
    """
    
    def __init__(self,
                 vocab_size=50257,  # GPT-2 BPE vocab size
                 embedding_dim=256,
                 num_layers=6,
                 num_heads=8,
                 ffn_dim=1024,
                 field_size=512,
                 max_seq_len=2048,
                 dropout=0.1,
                 use_checkpoint=False,
                 use_global_context=True,
                 global_context_interval=2,
                 device=None):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.use_checkpoint = use_checkpoint
        self.use_global_context = use_global_context
        self.global_context_interval = global_context_interval
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(embedding_dim, max_cache=max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            CausalFieldTransformerLayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                field_size=field_size,
                dropout=dropout,
                device=self.device
            )
            for _ in range(num_layers)
        ])
        
        # Global context modules (inserted every N layers)
        if use_global_context:
            num_gc = num_layers // global_context_interval
            self.global_context_modules = nn.ModuleList([
                GlobalContextModule(embedding_dim=embedding_dim, dropout=dropout)
                for _ in range(num_gc)
            ])
        else:
            self.global_context_modules = nn.ModuleList([])
        
        # Output
        self.norm = nn.LayerNorm(embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocab_size, bias=False)
        
        # Tie weights (common practice)
        self.output_projection.weight = self.token_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, labels=None, mask=None):
        """
        Forward pass.
        
        input_ids: (B, N) - token indices
        labels: (B, N) - target token indices (for training)
        mask: (B, N) - attention mask (optional)
        
        Returns: logits (B, N, vocab_size) and loss (if labels provided)
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        B, N = input_ids.shape
        
        # Embeddings
        x = self.token_embedding(input_ids)
        
        # Add positional encoding
        pos_enc = self.positional_encoding(N, input_ids.device)
        x = x + pos_enc.unsqueeze(0)
        x = self.dropout(x)
        
        # Transformer layers with global context
        gc_idx = 0
        for i, layer in enumerate(self.layers):
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, mask, use_reentrant=False
                )
            else:
                x = layer(x, mask)
            
            # Apply global context after every N layers
            if (self.use_global_context and
                (i + 1) % self.global_context_interval == 0 and
                gc_idx < len(self.global_context_modules)):
                x = self.global_context_modules[gc_idx](x)
                gc_idx += 1
        
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
    
    def generate(self,
                 input_ids,
                 max_new_tokens=100,
                 temperature=1.0,
                 top_k=50,
                 top_p=0.9,
                 repetition_penalty=1.2):
        """
        Generate text autoregressively.
        
        input_ids: (B, N) or (N,) - prompt tokens
        max_new_tokens: number of tokens to generate
        temperature: sampling temperature
        top_k: keep only top k tokens
        top_p: nucleus sampling threshold
        repetition_penalty: penalize repeated tokens
        """
        self.eval()
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Get logits for next token
                logits, _ = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Repetition penalty
                if repetition_penalty != 1.0 and generated.shape[1] > 1:
                    for token_id in set(generated[0, -50:].tolist()):
                        next_token_logits[0, token_id] /= repetition_penalty
                
                # Top-k filtering
                if top_k is not None and top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if max length reached
                if generated.shape[1] >= self.max_seq_len:
                    break
        
        return generated


if __name__ == '__main__':
    print("\nTesting Causal Field Transformer...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = CausalFieldTransformer(
        vocab_size=256,  # Char-level for quick test
        embedding_dim=256,
        num_layers=4,
        num_heads=8,
        field_size=512,
        device=device
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")
    
    # Test forward
    x = torch.randint(0, 256, (2, 100), device=device)
    y = torch.randint(0, 256, (2, 100), device=device)
    
    logits, loss = model(x, labels=y)
    
    print(f"Input: {x.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Loss: {loss.item():.3f}")
    
    # Test generation
    prompt = torch.randint(0, 256, (1, 10), device=device)
    generated = model.generate(prompt, max_new_tokens=50, temperature=0.8)
    
    print(f"Generated: {generated.shape}")
    print("✓ Causal Field Transformer works!")

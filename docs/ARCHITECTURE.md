# Field LLM Architecture

This document provides a detailed technical architecture for Field LLM implementation.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Field LLM                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input Tokens                                                │
│       ↓                                                      │
│  ┌─────────────────────────────────────┐                   │
│  │   Token Embedding Layer             │                   │
│  │   - Continuous semantic positions   │                   │
│  │   - Learnable position encoding     │                   │
│  └─────────────────────────────────────┘                   │
│       ↓                                                      │
│  ┌─────────────────────────────────────┐                   │
│  │   Field Transformer Layers (×N)     │                   │
│  │   ┌───────────────────────────────┐ │                   │
│  │   │  1. Token → Field Embedding   │ │                   │
│  │   │  2. Multi-Resolution Fields   │ │                   │
│  │   │  3. FFT-based Attention       │ │                   │
│  │   │  4. Field → Token Sampling    │ │                   │
│  │   │  5. Feed-Forward Network      │ │                   │
│  │   └───────────────────────────────┘ │                   │
│  └─────────────────────────────────────┘                   │
│       ↓                                                      │
│  ┌─────────────────────────────────────┐                   │
│  │   Continuous Memory (Optional)      │                   │
│  │   - Infinite context storage        │                   │
│  │   - Field-based retrieval           │                   │
│  └─────────────────────────────────────┘                   │
│       ↓                                                      │
│  ┌─────────────────────────────────────┐                   │
│  │   Output Head                       │                   │
│  │   - Field → Token projection        │                   │
│  │   - Vocabulary prediction           │                   │
│  └─────────────────────────────────────┘                   │
│       ↓                                                      │
│  Output Tokens                                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Grid Data Structure

The foundation of all field operations:

```python
class Grid2D:
    """2D density field with FFT support"""
    
    def __init__(self, size: Tuple[int, int], device: str = 'cuda'):
        self.width, self.height = size
        self.data = torch.zeros(size, device=device)
        
        # Precompute FFT plans for efficiency
        self.fft_plan = self._create_fft_plan()
    
    def add_gaussian(self, x: float, y: float, 
                     magnitude: float, sigma: float):
        """Add Gaussian blob at continuous position"""
        # Convert continuous position to grid coordinates
        gx, gy = self._to_grid_coords(x, y)
        
        # Compute Gaussian kernel
        kernel = self._gaussian_kernel(gx, gy, sigma)
        
        # Add to field
        self.data += magnitude * kernel
    
    def convolve(self, other: 'Grid2D') -> 'Grid2D':
        """FFT-based convolution (O(n log n))"""
        # Forward FFT
        fft_self = torch.fft.rfft2(self.data)
        fft_other = torch.fft.rfft2(other.data)
        
        # Multiply in frequency domain
        fft_result = fft_self * fft_other
        
        # Inverse FFT
        result_data = torch.fft.irfft2(fft_result)
        
        return Grid2D.from_data(result_data)
    
    def sample(self, x: float, y: float) -> float:
        """Sample field at continuous position (bilinear interpolation)"""
        gx, gy = self._to_grid_coords(x, y)
        return self._bilinear_interpolate(gx, gy)
```

### 2. Token ↔ Field Conversion

Critical for bridging discrete tokens and continuous fields:

```python
class TokenFieldConverter:
    """Convert between discrete tokens and continuous fields"""
    
    def __init__(self, grid_size: Tuple[int, int], 
                 embedding_dim: int,
                 spread_radius: float = 2.0):
        self.grid_size = grid_size
        self.embedding_dim = embedding_dim
        self.spread_radius = spread_radius
        
        # Learnable semantic position encoder
        self.position_encoder = nn.Linear(embedding_dim, 2)
    
    def tokens_to_field(self, 
                       token_embeddings: torch.Tensor,
                       positions: torch.Tensor) -> Grid2D:
        """
        Convert tokens to continuous field
        
        Args:
            token_embeddings: (batch, seq_len, embed_dim)
            positions: (batch, seq_len) - sequential positions
        
        Returns:
            field: Grid2D representing token density
        """
        batch_size, seq_len, embed_dim = token_embeddings.shape
        field = Grid2D(self.grid_size)
        
        for b in range(batch_size):
            for i in range(seq_len):
                # Compute continuous semantic position
                emb = token_embeddings[b, i]
                semantic_pos = self.position_encoder(emb)  # (2,)
                
                # Add positional bias (preserve sequence order)
                pos_bias = self._positional_bias(positions[b, i], seq_len)
                x = semantic_pos[0] + pos_bias[0]
                y = semantic_pos[1] + pos_bias[1]
                
                # Add Gaussian blob to field
                magnitude = torch.norm(emb)
                field.add_gaussian(x, y, magnitude, self.spread_radius)
        
        return field
    
    def field_to_tokens(self, 
                       field: Grid2D,
                       num_tokens: int,
                       positions: torch.Tensor) -> torch.Tensor:
        """
        Sample field to reconstruct tokens
        
        Args:
            field: Grid2D to sample from
            num_tokens: Number of tokens to generate
            positions: (batch, seq_len) - where to sample
        
        Returns:
            token_embeddings: (batch, seq_len, embed_dim)
        """
        batch_size = positions.shape[0]
        seq_len = positions.shape[1]
        
        token_embeddings = torch.zeros(
            batch_size, seq_len, self.embedding_dim
        )
        
        for b in range(batch_size):
            for i in range(seq_len):
                # Compute where to sample (inverse of tokens_to_field)
                pos_bias = self._positional_bias(positions[b, i], seq_len)
                
                # Sample field in neighborhood
                samples = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        x = pos_bias[0] + dx * self.spread_radius
                        y = pos_bias[1] + dy * self.spread_radius
                        samples.append(field.sample(x, y))
                
                # Reconstruct embedding from samples
                token_embeddings[b, i] = self._reconstruct_embedding(samples)
        
        return token_embeddings
    
    def _positional_bias(self, position: int, seq_len: int) -> Tuple[float, float]:
        """Convert sequential position to 2D coordinates"""
        # Map [0, seq_len) to grid coordinates
        # Use space-filling curve (e.g., Hilbert) for better locality
        t = position / seq_len  # Normalize to [0, 1)
        
        # Simple linear mapping (can be improved with Hilbert curve)
        x = (t * self.grid_size[0]) % self.grid_size[0]
        y = (t * self.grid_size[0]) // self.grid_size[0]
        
        return (x, y)
```

### 3. Field Attention Mechanism

The core innovation:

```python
class FieldAttention(nn.Module):
    """Field-based attention using FFT convolution"""
    
    def __init__(self, 
                 grid_size: Tuple[int, int],
                 embedding_dim: int,
                 num_heads: int = 8,
                 attention_radius: float = 10.0):
        super().__init__()
        
        self.grid_size = grid_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.attention_radius = attention_radius
        
        # Standard Q, K, V projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Token ↔ Field converter
        self.converter = TokenFieldConverter(grid_size, self.head_dim)
        
        # Learnable attention kernel
        self.attention_kernel = self._create_attention_kernel()
    
    def forward(self, 
                x: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        """
        Field-based attention
        
        Args:
            x: (batch, seq_len, embed_dim)
            positions: (batch, seq_len)
        
        Returns:
            output: (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # 1. Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, embed_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 3. Process each head with field attention
        outputs = []
        for h in range(self.num_heads):
            q_h = q[:, :, h, :]  # (batch, seq_len, head_dim)
            k_h = k[:, :, h, :]
            v_h = v[:, :, h, :]
            
            # Convert to fields
            q_field = self.converter.tokens_to_field(q_h, positions)
            k_field = self.converter.tokens_to_field(k_h, positions)
            v_field = self.converter.tokens_to_field(v_h, positions)
            
            # Compute attention via FFT convolution
            # attention_field = softmax(q_field ⊗ k_field)
            qk_field = q_field.convolve(k_field)
            attention_field = self._apply_attention_kernel(qk_field)
            attention_field = self._softmax_field(attention_field)
            
            # Apply attention to values
            output_field = attention_field.multiply(v_field)
            
            # Convert back to tokens
            output_h = self.converter.field_to_tokens(
                output_field, seq_len, positions
            )
            outputs.append(output_h)
        
        # 4. Concatenate heads
        output = torch.cat(outputs, dim=-1)  # (batch, seq_len, embed_dim)
        
        # 5. Output projection
        output = self.out_proj(output)
        
        return output
    
    def _create_attention_kernel(self) -> Grid2D:
        """Create learnable attention kernel (distance-based decay)"""
        kernel = Grid2D(self.grid_size)
        
        cx, cy = self.grid_size[0] / 2, self.grid_size[1] / 2
        
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                # Distance from center
                distance = math.sqrt((x - cx)**2 + (y - cy)**2)
                
                # Exponential decay
                weight = math.exp(-distance / self.attention_radius)
                
                kernel.data[x, y] = weight
        
        return kernel
    
    def _apply_attention_kernel(self, field: Grid2D) -> Grid2D:
        """Apply attention kernel via convolution"""
        return field.convolve(self.attention_kernel)
    
    def _softmax_field(self, field: Grid2D) -> Grid2D:
        """Apply softmax normalization to field"""
        # Normalize so field integrates to 1
        total = field.data.sum()
        normalized = Grid2D.from_data(field.data / (total + 1e-8))
        return normalized
```

### 4. Hierarchical Multi-Resolution Fields

For capturing both local and global context:

```python
class HierarchicalFieldAttention(nn.Module):
    """Multi-resolution field attention"""
    
    def __init__(self, 
                 embedding_dim: int,
                 num_heads: int = 8):
        super().__init__()
        
        # Three resolution levels
        self.global_attention = FieldAttention(
            grid_size=(64, 64),
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            attention_radius=32.0  # Long-range
        )
        
        self.medium_attention = FieldAttention(
            grid_size=(256, 256),
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            attention_radius=16.0  # Medium-range
        )
        
        self.local_attention = FieldAttention(
            grid_size=(1024, 1024),
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            attention_radius=4.0   # Short-range
        )
        
        # Learnable weights for combining resolutions
        self.global_weight = nn.Parameter(torch.tensor(0.2))
        self.medium_weight = nn.Parameter(torch.tensor(0.3))
        self.local_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Multi-resolution attention"""
        
        # Compute attention at each resolution
        global_output = self.global_attention(x, positions)
        medium_output = self.medium_attention(x, positions)
        local_output = self.local_attention(x, positions)
        
        # Weighted combination
        output = (
            self.global_weight * global_output +
            self.medium_weight * medium_output +
            self.local_weight * local_output
        )
        
        return output
```

### 5. Field Transformer Layer

Complete transformer layer with field attention:

```python
class FieldTransformerLayer(nn.Module):
    """Single Field Transformer layer"""
    
    def __init__(self,
                 embedding_dim: int,
                 num_heads: int = 8,
                 ffn_dim: int = 2048,
                 dropout: float = 0.1,
                 use_hierarchical: bool = True):
        super().__init__()
        
        # Field attention (or hierarchical)
        if use_hierarchical:
            self.attention = HierarchicalFieldAttention(
                embedding_dim, num_heads
            )
        else:
            self.attention = FieldAttention(
                grid_size=(1024, 1024),
                embedding_dim=embedding_dim,
                num_heads=num_heads
            )
        
        # Feed-forward network (standard)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embedding_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (batch, seq_len, embed_dim)
            positions: (batch, seq_len)
        """
        # Self-attention with residual
        attn_output = self.attention(self.norm1(x), positions)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output
        
        return x
```

### 6. Complete Field Transformer Model

```python
class FieldTransformer(nn.Module):
    """Complete Field Transformer model"""
    
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 8,
                 ffn_dim: int = 2048,
                 max_seq_len: int = 1_000_000,
                 dropout: float = 0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding (learnable)
        self.positional_encoding = nn.Embedding(max_seq_len, embedding_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            FieldTransformerLayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                use_hierarchical=True
            )
            for _ in range(num_layers)
        ])
        
        # Output head
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len) - not used (field handles naturally)
        
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        # Embed tokens
        x = self.token_embedding(input_ids)  # (batch, seq_len, embed_dim)
        
        # Add positional encoding
        pos_emb = self.positional_encoding(positions)
        x = x + pos_emb
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, positions)
        
        # Output projection
        x = self.output_norm(x)
        logits = self.output_projection(x)
        
        return logits
    
    def generate(self, 
                 input_ids: torch.Tensor,
                 max_new_tokens: int = 100,
                 temperature: float = 1.0) -> torch.Tensor:
        """
        Autoregressive generation
        
        Args:
            input_ids: (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
        
        Returns:
            generated_ids: (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Forward pass
            logits = self.forward(input_ids)  # (batch, seq_len, vocab_size)
            
            # Get next token logits
            next_logits = logits[:, -1, :] / temperature
            
            # Sample next token
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
```

## Memory Analysis

### Standard Transformer

For sequence length `n` and embedding dimension `d`:

```
Attention matrix: n × n × d × 4 bytes
For n=1M, d=768: 1M × 1M × 768 × 4 = 3 TB
```

### Field Transformer

For grid size `g × g` and embedding dimension `d`:

```
Query field: g × g × 4 bytes
Key field: g × g × 4 bytes
Value field: g × g × 4 bytes
Total: 3 × g² × 4 bytes

For g=1024: 3 × 1024² × 4 = 12 MB
```

**Compression ratio**: 3 TB / 12 MB = **250,000×**

## Computational Complexity

### Standard Attention

```
O(n² × d) per layer
For n=1M, d=768: 768 trillion operations
```

### Field Attention

```
Token → Field: O(n × g²)
FFT Convolution: O(g² × log(g²))
Field → Token: O(n × g²)
Total: O(n × g² + g² × log(g²))

For n=1M, g=1024:
  n × g²: 1M × 1M = 1 trillion
  g² × log(g²): 1M × 20 = 20 million
  Total: ~1 trillion (dominated by token conversion)

But with fixed g, this becomes O(n) linear!
```

## Implementation Considerations

### 1. GPU Optimization

- Use CUDA kernels for field operations
- Reuse FFT plans across batches
- Fuse token→field and field→token operations

### 2. Numerical Stability

- Use log-space for attention softmax
- Normalize fields to prevent overflow
- Gradient clipping for field parameters

### 3. Training Strategies

- **Curriculum learning**: Start with small grids, gradually increase
- **Mixed precision**: Use FP16 for fields, FP32 for critical ops
- **Gradient checkpointing**: Save memory by recomputing fields

### 4. Inference Optimization

- **KV caching**: Store field representations, not token pairs
- **Quantization**: Use INT8 for field values
- **Sparse fields**: Only store non-zero regions

## Next Steps

1. Implement `Grid2D` with CUDA kernels
2. Build `TokenFieldConverter` with learnable positions
3. Create `FieldAttention` module
4. Train on copy task to validate
5. Scale to real language data

See `examples/` for concrete implementations.

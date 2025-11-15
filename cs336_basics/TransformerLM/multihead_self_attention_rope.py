import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
import math


class MultiHeadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention module.
    
    This implementation follows Vaswani et al., 2017 with causal masking
    to prevent positions from attending to subsequent positions.
    Optionally supports RoPE (Rotary Position Embeddings).
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        max_seq_len: int = None, 
        theta: float = None,
        device=None,
        dtype=None
    ):
        """
        Args:
            d_model: Dimensionality of the Transformer block inputs/outputs
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length (for RoPE precomputation)
            theta: RoPE base parameter (if None, RoPE is disabled)
            device: Device to place parameters on
            dtype: Data type for parameters
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # Dimension per head
        self.d_v = d_model // num_heads # Same as d_k per the paper
        
        # Combined projections for all heads (more efficient than separate projections)
        self.q_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        
        # Output projection
        self.o_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        
        # Scaling factor for scaled dot-product attention
        self.scale = math.sqrt(self.d_k)
        
        # RoPE parameters
        self.use_rope = theta is not None
        if self.use_rope:
            self.theta = theta
            self.max_seq_len = max_seq_len
            self._precompute_rope_freqs(device=device)  # ← Передаємо device
    
    def _precompute_rope_freqs(self, device=None):
        """Precompute RoPE rotation frequencies."""
        # Compute frequencies for each dimension pair
        # freq_i = 1 / (theta ^ (2i / d_k)) for i in [0, d_k/2)
        dim_indices = torch.arange(0, self.d_k, 2, dtype=torch.float32, device=device)
        freqs = 1.0 / (self.theta ** (dim_indices / self.d_k))
        
        # Register as buffer so it moves with the model
        self.register_buffer('rope_freqs', freqs, persistent=False)
    
    def _apply_rope(
        self, 
        x: Float[Tensor, "... seq_len num_heads d_k"],
        positions: Int[Tensor, "... seq_len"] = None
    ) -> Float[Tensor, "... seq_len num_heads d_k"]:
        """
        Apply RoPE to input tensor.
        
        Args:
            x: Input tensor (..., seq_len, num_heads, d_k)
            positions: Position indices (..., seq_len)
        
        Returns:
            Rotated tensor of same shape
        """
        seq_len = x.shape[-3]
        
        # Default to sequential positions if not provided
        if positions is None:
            positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
            # Expand to match batch dimensions
            for _ in range(len(x.shape) - 3):
                positions = positions.unsqueeze(0)
        
        # Compute position * frequency for each dimension
        # positions: (..., seq_len), freqs: (d_k/2,)
        # Result: (..., seq_len, d_k/2)
        pos_freqs = positions.unsqueeze(-1).float() * self.rope_freqs.unsqueeze(0)
        
        # Compute cos and sin
        cos_pos = torch.cos(pos_freqs)  # (..., seq_len, d_k/2)
        sin_pos = torch.sin(pos_freqs)  # (..., seq_len, d_k/2)
        
        # Add head dimension
        cos_pos = cos_pos.unsqueeze(-2)  # (..., seq_len, 1, d_k/2)
        sin_pos = sin_pos.unsqueeze(-2)  # (..., seq_len, 1, d_k/2)
        
        # Split x into even and odd indices
        x_even = x[..., 0::2]  # (..., seq_len, num_heads, d_k/2)
        x_odd = x[..., 1::2]   # (..., seq_len, num_heads, d_k/2)
        
        # Apply rotation: [x_even, x_odd] -> [x_even*cos - x_odd*sin, x_even*sin + x_odd*cos]
        x_rotated_even = x_even * cos_pos - x_odd * sin_pos
        x_rotated_odd = x_even * sin_pos + x_odd * cos_pos
        
        # Interleave back together
        x_rotated = torch.empty_like(x)
        x_rotated[..., 0::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd
        
        return x_rotated
    
    def forward(
        self, 
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] = None
    ) -> Float[Tensor, "... seq_len d_model"]:
        """
        Forward pass of causal multi-head self-attention.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            token_positions: Optional position indices for RoPE (..., seq_len)
        
        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        batch_shape = x.shape[:-2]
        seq_len = x.shape[-2]
        
        # Project to Q, K, V: (..., seq_len, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape and transpose for multi-head attention
        # From: (..., seq_len, d_model) 
        # To: (..., seq_len, num_heads, d_k)
        Q = Q.view(*batch_shape, seq_len, self.num_heads, self.d_k)
        K = K.view(*batch_shape, seq_len, self.num_heads, self.d_k)
        V = V.view(*batch_shape, seq_len, self.num_heads, self.d_v)
        
        # Apply RoPE to Q and K if enabled
        if self.use_rope:
            Q = self._apply_rope(Q, token_positions)
            K = self._apply_rope(K, token_positions)
        
        # Transpose to (..., num_heads, seq_len, d_k)
        Q = Q.transpose(-3, -2)
        K = K.transpose(-3, -2)
        V = V.transpose(-3, -2)
        
        # Scaled dot-product attention
        # Q @ K^T: (..., num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply causal mask (prevent attending to future positions)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values: (..., num_heads, seq_len, d_v)
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads: (..., seq_len, d_model)
        attn_output = attn_output.transpose(-3, -2).contiguous()
        attn_output = attn_output.view(*batch_shape, seq_len, self.d_model)
        
        # Final output projection
        output = self.o_proj(attn_output)
        
        return output
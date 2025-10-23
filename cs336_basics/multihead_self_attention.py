import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
import math


class MultiHeadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention module.
    
    This implementation follows Vaswani et al., 2017 with causal masking
    to prevent positions from attending to subsequent positions.
    """
    
    def __init__(self, d_model: int, num_heads: int):
        """
        Args:
            d_model: Dimensionality of the Transformer block inputs/outputs
            num_heads: Number of attention heads
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        self.d_v = d_model // num_heads  # Same as d_k per the paper
        
        # Combined projections for all heads (more efficient than separate projections)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Scaling factor for scaled dot-product attention
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self, 
        x: Float[Tensor, "... seq_len d_model"]
    ) -> Float[Tensor, "... seq_len d_model"]:
        """
        Forward pass of causal multi-head self-attention.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model)
        
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
        # To: (..., num_heads, seq_len, d_k)
        Q = Q.view(*batch_shape, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        K = K.view(*batch_shape, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        V = V.view(*batch_shape, seq_len, self.num_heads, self.d_v).transpose(-3, -2)
        
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
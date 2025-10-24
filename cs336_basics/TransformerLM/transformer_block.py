import torch
import torch.nn as nn
from cs336_basics.TransformerLM.multihead_self_attention_rope import MultiHeadSelfAttention
from cs336_basics.TransformerLM.positionwise_feedforward import SwiGLU
from cs336_basics.TransformerLM.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with RMSNorm.
    
    Architecture (per sublayer):
        1. Apply RMSNorm
        2. Apply main operation (MultiHeadSelfAttention or FeedForward)
        3. Add residual connection
    
    The full block consists of two sublayers:
        - Multi-head self-attention sublayer
        - Position-wise feed-forward sublayer
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        eps: float = 1e-5,
        bias: bool = False,
        max_seq_len: int = None,
        theta: float = None
    ):
        """
        Initialize the Transformer block.
        
        Args:
            d_model: Dimensionality of the Transformer block inputs/outputs
            num_heads: Number of heads to use in multi-head self-attention
            d_ff: Dimensionality of the position-wise feed-forward inner layer
            eps: Epsilon for RMSNorm (default: 1e-5)
            bias: Whether to use bias in feed-forward layers (default: False)
            max_seq_len: Maximum sequence length (for RoPE, if used)
            theta: RoPE base parameter (if None, RoPE is disabled)
        """
        super().__init__()
        
        # First sublayer: Multi-head self-attention
        self.norm1 = RMSNorm(d_model, eps=eps)
        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta
        )
        
        # Second sublayer: Position-wise feed-forward network
        self.norm2 = RMSNorm(d_model, eps=eps)
        self.feed_forward = SwiGLU(d_model=d_model, d_ff=d_ff, bias=bias)
    
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass of the Transformer block.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            token_positions: Optional position indices for RoPE (..., seq_len)
        
        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        # First sublayer: Multi-head self-attention with residual
        # y = x + MultiHeadSelfAttention(RMSNorm(x))
        x = x + self.attention(self.norm1(x), token_positions=token_positions)
        
        # Second sublayer: Feed-forward network with residual
        # y = x + FeedForward(RMSNorm(x))
        x = x + self.feed_forward(self.norm2(x))
        
        return x
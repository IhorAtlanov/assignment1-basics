import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network
    
    Implements the SwiGLU variant of the feed-forward network, which uses
    SiLU (Swish) activation and Gated Linear Unit (GLU).
    
    Architecture:
        FFN_SwiGLU(x) = (SiLU(xW1) ⊙ xW3)W2
    
    Where:
        - SiLU(x) = x * sigmoid(x)
        - ⊙ denotes element-wise multiplication (gating)
        - W1, W3: project from d_model to d_ff
        - W2: project from d_ff back to d_model
    
    Args:
        d_model: Input/output dimension
        d_ff: Inner feed-forward dimension (optional, defaults to ~8/3 * d_model)
        bias: Whether to use bias in linear layers (default: False)
    """
    
    def __init__(self, d_model: int, d_ff: int = None, bias: bool = False):
        super().__init__()
        
        # Calculate d_ff as approximately (8/3) * d_model
        if d_ff is None:
            d_ff = int(8 * d_model / 3)
        
        # Round to nearest multiple of 64 for hardware efficiency
        d_ff = ((d_ff + 63) // 64) * 64
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Three linear projections:
        # W1: for SiLU activation path
        # W3: for gating path
        # W2: output projection
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w3 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SwiGLU FFN
        
        Args:
            x: Input tensor of shape (..., d_model)
        
        Returns:
            Output tensor of shape (..., d_model)
        """
        # SiLU activation path: x * sigmoid(x)
        # Using torch.sigmoid for numerical stability as specified
        swish = self.w1(x) * torch.sigmoid(self.w1(x))
        
        # Gating path
        gate = self.w3(x)
        
        # Element-wise multiplication (gating mechanism)
        gated = swish * gate
        
        # Output projection
        output = self.w2(gated)
        
        return output
import torch
import torch.nn as nn

class Liner(nn.Module):
  def __init__(self, in_features: int , out_features: int, device: torch.device | None=None, dtype: torch.dtype| None=None):
    super().__init__()  # CRITICAL: Must call parent constructor
    # Create weight parameter W with shape (in_features, out_features)
    # Store as W (not W transpose) for memory ordering reasons
    self.W = nn.Parameter(
        torch.empty(in_features, out_features, device=device, dtype=dtype)
    )
    
    # Initialize weights using truncated normal distribution
    nn.init.trunc_normal_(self.W)
  
  def forward(self, x: torch.Tensor)-> torch.Tensor:
    return x @ self.W
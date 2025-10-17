import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module.
        
        Args:
            d_model: Hidden dimension of the model
            eps: Epsilon value for numerical stability (default: 1e-5)
            device: Device to store the parameters on (default: None)
            dtype: Data type of the parameters (default: None)
        """
        super().__init__()
        self.eps = eps
        
        # Learnable scale parameter (gain)
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor and apply RMSNorm.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_model)
            
        Returns:
            Normalized tensor of the same shape as input
        """
        # Store original dtype for later downcasting
        original_dtype = x.dtype
        
        # Upcast to float32 for numerical stability
        x = x.float()
        
        # Compute RMS: sqrt(mean(x^2))
        # Shape: (batch_size, sequence_length, 1)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize by RMS
        x_norm = x / rms
        
        # Downcast back to original dtype
        x_norm = x_norm.to(original_dtype)
        
        # Apply learnable scale (weight)
        # Broadcasting: (batch_size, seq_len, d_model) * (d_model,)
        return x_norm * self.weight
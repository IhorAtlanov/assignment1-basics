import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.
        
        Args:
            theta: float Θ value for the RoPE
            d_k: int dimension of query and key vectors
            max_seq_len: int Maximum sequence length that will be inputted
            device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # Precompute frequency values for each dimension pair
        # freq_i = 1 / (theta^(2i/d_k)) for i in [0, d_k/2)
        indices = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        freqs = 1.0 / (theta ** (indices / d_k))
        
        # Precompute position embeddings for all positions
        # positions: [max_seq_len]
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        
        # Compute m * freq for all positions and frequencies
        # Shape: [max_seq_len, d_k/2]
        freqs_pos = torch.outer(positions, freqs)
        
        # Precompute cos and sin values
        # Shape: [max_seq_len, d_k/2]
        cos = torch.cos(freqs_pos)
        sin = torch.sin(freqs_pos)
        
        # Register as buffers so they move with the module
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor and apply rotary positional embedding.
        
        Args:
            x: torch.Tensor of shape (..., seq_len, d_k)
            token_positions: torch.Tensor of shape (..., seq_len) specifying token positions
            
        Returns:
            torch.Tensor of shape (..., seq_len, d_k)
        """
        # Get the sequence length from x
        seq_len = x.shape[-2]
        
        # Split x into even and odd indices: [x0, x1, x2, x3, ...] -> [x0, x2, ...] and [x1, x3, ...]
        x_even = x[..., 0::2]  # Shape: (..., seq_len, d_k/2)
        x_odd = x[..., 1::2]   # Shape: (..., seq_len, d_k/2)
        
        # Select cos and sin values based on token positions
        # token_positions: (..., seq_len) with values in [0, max_seq_len)
        # We need to index into cos and sin which have shape [max_seq_len, d_k/2]
        
        # Flatten batch dimensions for indexing
        original_shape = x.shape
        token_positions_flat = token_positions.reshape(-1)  # [batch_size * seq_len]
        
        # Index into cos and sin
        cos_selected = self.cos[token_positions_flat]  # [batch_size * seq_len, d_k/2]
        sin_selected = self.sin[token_positions_flat]  # [batch_size * seq_len, d_k/2]
        
        # Reshape back to match x's batch dimensions
        batch_dims = original_shape[:-2]
        cos_selected = cos_selected.reshape(*batch_dims, seq_len, -1)  # (..., seq_len, d_k/2)
        sin_selected = sin_selected.reshape(*batch_dims, seq_len, -1)  # (..., seq_len, d_k/2)
        
        # Apply rotation
        # [x0, x1, x2, x3, ...] -> [x0*cos - x1*sin, x0*sin + x1*cos, x2*cos - x3*sin, ...]
        x_even_rotated = x_even * cos_selected - x_odd * sin_selected
        x_odd_rotated = x_even * sin_selected + x_odd * cos_selected
        
        # Interleave the results back
        x_out = torch.empty_like(x)
        x_out[..., 0::2] = x_even_rotated
        x_out[..., 1::2] = x_odd_rotated
        
        return x_out
import torch
import torch.nn as nn
from cs336_basics.TransformerLM.embedding import Embedding
from cs336_basics.TransformerLM.transformer_block import TransformerBlock
from cs336_basics.TransformerLM.rmsnorm import RMSNorm
from cs336_basics.TransformerLM.liner import Liner


class TransformerLM(nn.Module):
    """
    Transformer Language Model.
    
    Architecture:
        1. Token embeddings
        2. Stack of Transformer blocks
        3. Final RMSNorm
        4. Output projection to vocabulary (unembedding)
    
    This implementation uses RoPE for positional information instead of
    learned position embeddings.
    """
    
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        eps: float = 1e-5,
        bias: bool = False,
        theta: float = 10000.0,
        device: torch.device = None,
        dtype: torch.dtype = None
    ):
        """
        Initialize the Transformer Language Model.
        
        Args:
            vocab_size: Size of the vocabulary
            context_length: Maximum context length (max sequence length)
            d_model: Dimensionality of the model (embedding dimension)
            num_layers: Number of Transformer blocks
            num_heads: Number of attention heads in each block
            d_ff: Dimensionality of the feed-forward inner layer
            eps: Epsilon for RMSNorm (default: 1e-5)
            bias: Whether to use bias in feed-forward layers (default: False)
            theta: RoPE base parameter (default: 10000.0)
            device: Device to place parameters on
            dtype: Data type for parameters
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Token embedding layer
        self.token_embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype
        )
        
        # Stack of Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                eps=eps,
                bias=bias,
                max_seq_len=context_length,
                theta=theta
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        
        # Output projection (unembedding) - projects to vocabulary
        self.output_proj = Liner(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype
        )
    
    def forward(
        self,
        token_ids: torch.Tensor,
        token_positions: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass of the Transformer Language Model.
        
        Args:
            token_ids: Input token indices of shape (batch_size, seq_len)
            token_positions: Optional position indices for RoPE (batch_size, seq_len)
                           If None, sequential positions [0, 1, 2, ...] are used
        
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape
        
        # Create default sequential positions if not provided
        if token_positions is None:
            token_positions = torch.arange(
                seq_len,
                device=token_ids.device,
                dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1)
        
        # Token embeddings: (batch_size, seq_len, d_model)
        x = self.token_embedding(token_ids)
        
        # Pass through each Transformer block
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        
        # Final normalization
        x = self.norm(x)
        
        # Project to vocabulary: (batch_size, seq_len, vocab_size)
        logits = self.output_proj(x)
        
        return logits
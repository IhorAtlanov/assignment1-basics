import torch
import torch.nn as nn


class Embedding(nn.Module): 
    def __init__( self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Initialize the embedding matrix as a parameter
        # Shape: (num_embeddings, embedding_dim)
        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype
            )
        )
        
        # Initialize weights using truncated normal distribution
        nn.init.trunc_normal_(self.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Perform embedding lookup using indexing
        # This is equivalent to a one-hot encoding followed by matrix multiplication
        return self.weight[token_ids]
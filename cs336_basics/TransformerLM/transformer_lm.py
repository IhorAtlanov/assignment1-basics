import torch
import torch.nn as nn
from cs336_basics.TransformerLM.embedding import Embedding
from cs336_basics.TransformerLM.transformer_block import TransformerBlock
from cs336_basics.TransformerLM.rmsnorm import RMSNorm
from cs336_basics.TransformerLM.liner import Liner


class TransformerLM(nn.Module):
    """
    Transformer Language Model.
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
        """Initialize the Transformer Language Model."""
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
                theta=theta,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        
        # Output projection (unembedding)
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
        batch_size, seq_len = token_ids.shape
        
        if token_positions is None:
            token_positions = torch.arange(
                seq_len,
                device=token_ids.device,
                dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1)
        
        x = self.token_embedding(token_ids)
        
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        
        x = self.norm(x)
        logits = self.output_proj(x)
        
        return logits
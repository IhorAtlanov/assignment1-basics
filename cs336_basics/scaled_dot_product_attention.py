import torch
import math
from cs336_basics.softmax import softmax


def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Compute scaled dot-product attention.
    
    Args:
        query: Tensor of shape (batch_size, ..., seq_len, d_k)
        key: Tensor of shape (batch_size, ..., seq_len, d_k)
        value: Tensor of shape (batch_size, ..., seq_len, d_v)
        mask: Optional boolean tensor of shape (seq_len, seq_len)
              True = attend, False = mask out
    
    Returns:
        output: Tensor of shape (batch_size, ..., seq_len, d_v)
        attention_weights: Tensor of shape (batch_size, ..., seq_len, seq_len)
    """
    # Get d_k from the last dimension of query
    d_k = query.shape[-1]
    
    # Compute attention scores: Q @ K^T / sqrt(d_k)
    # query: (..., seq_len, d_k)
    # key.transpose(-2, -1): (..., d_k, seq_len)
    # scores: (..., seq_len, seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        # mask shape: (seq_len, seq_len)
        # scores shape: (batch_size, ..., seq_len, seq_len)
        # Broadcast mask to match scores shape
        # Set masked positions (False) to large negative value
        scores = scores.masked_fill(~mask, float('-inf'))
    
    # Apply softmax to get attention weights
    # softmax along the last dimension (key dimension)
    attention_weights = softmax(scores, dim=-1)
    
    # Handle NaN values that can occur when entire row is masked
    # (all -inf before softmax results in NaN)
    attention_weights = torch.where(
        torch.isnan(attention_weights),
        torch.zeros_like(attention_weights),
        attention_weights
    )
    
    # Apply attention weights to values
    # attention_weights: (..., seq_len, seq_len)
    # value: (..., seq_len, d_v)
    # output: (..., seq_len, d_v)
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights
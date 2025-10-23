import torch

def softmax(tensor, dim):
    """
    Apply softmax operation on a tensor along a specified dimension.
    
    Args:
        tensor: Input tensor of any shape
        dim: The dimension along which to apply softmax
        
    Returns:
        Output tensor with the same shape as input, with softmax applied along dim
    """
    # Step 1: Subtract max for numerical stability
    # keepdim=True maintains the dimension for broadcasting
    max_vals = tensor.max(dim=dim, keepdim=True)[0]
    shifted = tensor - max_vals
    
    # Step 2: Compute exponentials
    exp_vals = torch.exp(shifted)
    
    # Step 3: Sum along the specified dimension
    sum_exp = exp_vals.sum(dim=dim, keepdim=True)
    
    # Step 4: Normalize by dividing by the sum
    softmax_output = exp_vals / sum_exp
    
    return softmax_output
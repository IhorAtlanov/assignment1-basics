import torch
import torch.nn.functional as F


def run_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute cross entropy loss with numerical stability.
    
    The cross entropy loss is: ℓ = -log(softmax(logits)[target])
    
    With numerical stability:
    1. Subtract max for stability: logits' = logits - max(logits)
    2. softmax(logits')[target] = exp(logits'[target]) / sum(exp(logits'))
    3. log(softmax(logits')[target]) = logits'[target] - log(sum(exp(logits')))
    4. Cross entropy: ℓ = -logits'[target] + log(sum(exp(logits')))
    
    Args:
        logits: Tensor of shape (..., vocab_size) containing unnormalized log probabilities
        targets: Tensor of shape (...) containing target indices in range [0, vocab_size)
    
    Returns:
        Scalar tensor containing the average cross entropy loss across all batch dimensions
    """
    # logits shape: (..., vocab_size) where ... represents any number of batch dimensions
    # targets shape: (...) with integer indices
    
    # Step 1: Subtract max for numerical stability
    # We subtract along the vocabulary dimension (last dimension)
    max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
    logits_stable = logits - max_logits  # Shape: (..., vocab_size)
    
    # Step 2: Compute log(sum(exp(logits_stable)))
    # This is the log-sum-exp trick result
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_stable), dim=-1))  # Shape: (...)
    
    # Step 3: Extract the logit values at the target indices
    # We need to gather the logit values corresponding to each target
    # targets has shape (...), logits_stable has shape (..., vocab_size)
    
    # Flatten batch dimensions for easier indexing
    batch_shape = targets.shape
    num_elements = targets.numel()
    
    # Reshape logits_stable to (num_elements, vocab_size)
    logits_flat = logits_stable.view(num_elements, -1)
    
    # Reshape targets to (num_elements,)
    targets_flat = targets.view(num_elements)
    
    # Gather the target logits: for each batch element, get logits[target_index]
    target_logits = logits_flat[torch.arange(num_elements, device=logits.device), targets_flat]
    
    # Reshape back to original batch shape
    target_logits = target_logits.view(batch_shape)
    
    # Step 4: Compute cross entropy for each element
    # ℓ = -log(softmax(logits)[target])
    #   = -(logits_stable[target] - log_sum_exp)
    #   = log_sum_exp - logits_stable[target]
    cross_entropy_per_element = log_sum_exp - target_logits  # Shape: (...)
    
    # Step 5: Average across all batch dimensions
    return cross_entropy_per_element.mean()
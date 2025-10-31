import torch
from collections.abc import Iterable

def clip_grad_l2_(parameters: Iterable[torch.nn.Parameter],
                  max_l2_norm: float,
                  eps: float = 1e-6) -> float:
    """
    Clip gradients of the given parameters so that the total L2 norm <= max_l2_norm.
    Modifies gradients in-place.

    Returns:
        total_norm (float): L2 norm of all gradients BEFORE clipping.
    """
    params = list(parameters)
    grads = [p.grad for p in params if p.grad is not None]

    if len(grads) == 0:
        return 0.0

    total_sq = torch.tensor(0.0, device=grads[0].device, dtype=torch.float32)

    for g in grads:
        if g.is_sparse:
            # For sparse grads, use values (coalesce to be safe)
            vals = g.coalesce().values().detach().to(torch.float32)
            total_sq = total_sq + vals.norm(2).pow(2)
        else:
            # detach and compute in float32 for numeric stability (esp. mixed precision)
            d = g.detach().to(torch.float32)
            total_sq = total_sq + d.norm(2).pow(2)

    total_norm = torch.sqrt(total_sq).item()  # Python float

    # compute coefficient on the same device/dtype as grads if we'll use it in-place
    clip_coef = max_l2_norm / (total_norm + eps)

    if clip_coef < 1.0:
        # convert coef to tensor so in-place ops with CUDA float16 grads work safely
        coef_t = torch.tensor(clip_coef, device=grads[0].device, dtype=grads[0].dtype)
        for g in grads:
            if g.is_sparse:
                # scale sparse values in-place
                g._values().mul_(coef_t.to(g._values().dtype))
            else:
                g.mul_(coef_t.to(g.dtype))

    return total_norm

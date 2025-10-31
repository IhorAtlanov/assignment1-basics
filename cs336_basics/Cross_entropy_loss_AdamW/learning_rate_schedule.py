import math

def run_get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    """
    Return learning rate at iteration `it` following:
      - linear warmup for 0 <= t <= warmup_iters (t/warmup_iters * max_learning_rate)
      - cosine annealing for warmup_iters < t <= cosine_cycle_iters:
            alpha_t = alpha_min + 0.5*(alpha_max - alpha_min)*(1 + cos(pi * progress))
        where progress = (t - warmup_iters) / (cosine_cycle_iters - warmup_iters)
      - constant alpha_min for t > cosine_cycle_iters

    Parameters match the test names:
      it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters
    """
    t = it
    alpha_max = float(max_learning_rate)
    alpha_min = float(min_learning_rate)
    Tw = float(warmup_iters)
    Tc_end = float(cosine_cycle_iters)

    if t < 0:
        raise ValueError("t must be non-negative")

    # Warmup (include t == Tw)
    if Tw > 0 and t <= Tw:
        return alpha_max * (t / Tw)

    # If cosine end is not after warmup, just return alpha_min
    if Tc_end <= Tw:
        return alpha_min

    # Cosine annealing (warmup_iters < t <= cosine_cycle_iters)
    if t <= Tc_end:
        L = Tc_end - Tw
        progress = (t - Tw) / L  # in (0, 1]
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + math.cos(math.pi * progress))

    # After cosine end
    return alpha_min
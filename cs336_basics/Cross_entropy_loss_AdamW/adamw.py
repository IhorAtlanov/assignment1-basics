import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    Implements AdamW optimizer (Adam with decoupled weight decay).
    
    Arguments:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing running averages of gradient
               and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay: weight decay coefficient (default: 0.01)
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Arguments:
            closure: A closure that reevaluates the model and returns the loss (optional)
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                m, v = state['m'], state['v']
                state['step'] += 1
                
                # Update biased first moment estimate
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second raw moment estimate
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute bias-corrected first moment estimate
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute step size
                step_size = lr / bias_correction1
                
                # Compute bias-corrected second moment estimate (denominator)
                denom = (v.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                
                # Update parameters (Adam update)
                p.addcdiv_(m, denom, value=-step_size)
                
                # Decoupled weight decay (AdamW)
                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)
        
        return loss
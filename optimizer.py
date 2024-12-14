from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                

                # State should be stored in this dictionary
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros.like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                exp_avg,exp_avg_sq = state["exp_avg"],state["exp_avg_sq"]
                

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta1,beta2 =group["betas"]
                # Update first and second moments of the gradients
                self.state["step"]+=1
                exp_avg.mul_(beta1).add_(grad,alpha = 1-beta1)
                exp_avg_sq.mul_(beta2).add_(grad,grad,value = 1-beta2)
                
                # Bias correction
                bias_crr1 = 1 - beta1**state["step"]
                bias_crr2 = 1- beta2 **state["step"]
                
                crr_exp_avg = exp_avg / bias_crr1
                crr_exp_avg_sq = exp_avg_sq / bias_crr2
                
                lr = group["lr"]
                denomo = crr_exp_avg_sq.sqrt().add_(group["eps"])
                update = (crr_exp_avg / denomo + p.data * group["weight_decay"]) * (-lr)
                p.data = p.data+update
                
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980

                # Update parameters

                # Add weight decay after the main gradient-based updates.
                
                
                # Please note that the learning rate should be incorporated into this update.

        return loss
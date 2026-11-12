"""Rectified Adam.

Liu et al., "On the Variance of the Adaptive Learning Rate and Beyond",
ICLR 2020.

This is the implementation the published runs used, with the deprecated
in-place call signatures updated. ``addcmul_(value, t1, t2)`` and
``add_(value, t)`` were removed after PyTorch 1.5; the keyword forms
``addcmul_(t1, t2, value=...)`` and ``add_(t, alpha=...)`` are used instead and
are numerically identical.
"""

from __future__ import annotations

import math

import torch
from torch.optim.optimizer import Optimizer


class RAdam(Optimizer):
    """Adam with a rectified variance term.

    Args:
        params: iterable of parameters or parameter groups.
        lr: learning rate.
        betas: coefficients for the running averages.
        eps: term added to the denominator for numerical stability.
        weight_decay: L2 penalty, applied directly to the parameters.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # cache of (step, N_sma, step_size) keyed by step % 10
        self.buffer = [[None, None, None] for _ in range(10)]
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    msg = "RAdam does not support sparse gradients"
                    raise RuntimeError(msg)

                p_data_fp32 = p.data.float()
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state["step"] += 1
                buffered = self.buffer[int(state["step"] % 10)]
                if state["step"] == buffered[0]:
                    n_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    n_sma_max = 2 / (1 - beta2) - 1
                    n_sma = n_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    buffered[1] = n_sma

                    # conservative: this is an approximation
                    if n_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (n_sma - 4)
                            / (n_sma_max - 4)
                            * (n_sma - 2)
                            / n_sma
                            * n_sma_max
                            / (n_sma_max - 2)
                        ) / (1 - beta1 ** state["step"])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state["step"])
                    buffered[2] = step_size

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(
                        p_data_fp32, alpha=-group["weight_decay"] * group["lr"]
                    )

                if n_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_fp32.addcdiv_(
                        exp_avg, denom, value=-step_size * group["lr"]
                    )
                else:
                    p_data_fp32.add_(exp_avg, alpha=-step_size * group["lr"])

                p.data.copy_(p_data_fp32)

        return loss

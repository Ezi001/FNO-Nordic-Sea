"""Numerical integration helpers for CFO velocity fields (PyTorch version)."""

from __future__ import annotations

from typing import Any, Optional

import torch


def euler_step(
    state: Any,
    x0: torch.Tensor,
    t: torch.Tensor,
    condition: Optional[torch.Tensor],
    delta_t: float | torch.Tensor,
) -> torch.Tensor:
    """Advance one explicit Euler step."""
    velocity = state.model(x0, t, condition)
    return x0 + delta_t * velocity


def rk4_step(
    state: Any,
    x0: torch.Tensor,
    t: torch.Tensor,
    condition: Optional[torch.Tensor],
    delta_t: float | torch.Tensor,
) -> torch.Tensor:
    """Advance one classical Runge-Kutta (RK4) step."""
    k1 = state.model(x0, t, condition)
    k2 = state.model(x0 + 0.5 * delta_t * k1, t + 0.5 * delta_t, condition)
    k3 = state.model(x0 + 0.5 * delta_t * k2, t + 0.5 * delta_t, condition)
    k4 = state.model(x0 + delta_t * k3, t + delta_t, condition)
    return x0 + (delta_t / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def heun_step(
    state: Any,
    x0: torch.Tensor,
    t: torch.Tensor,
    condition: Optional[torch.Tensor],
    delta_t: float | torch.Tensor,
) -> torch.Tensor:
    """Advance one Heun (improved Euler) step."""
    k1 = state.model(x0, t, condition)
    x_predictor = x0 + delta_t * k1
    k2 = state.model(x_predictor, t + delta_t, condition)
    return x0 + 0.5 * delta_t * (k1 + k2)


INTEGRATOR_STEP_FNS = {
    "Euler": euler_step,
    "RK4": rk4_step,
    "Heun": heun_step,
}
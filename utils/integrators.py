"""Numerical integration helpers for CFO velocity fields (PyTorch version)."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

# def _velocity(self, x: Tensor, t: Tensor, condition: Tensor | None = None) -> Tensor:
#         if not torch.is_tensor(t):
#             t = torch.as_tensor(t, device=x.device, dtype=x.dtype)
#         if t.ndim == 0:
#             t = torch.full((x.shape[0],), t.item(), device=x.device, dtype=x.dtype)
#         return self._model_apply(x, t, condition)

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

    # def _euler_step(
    #     self,
    #     x: Tensor,
    #     t: Tensor,
    #     condition: Tensor | None,
    #     delta_t: Tensor,
    # ) -> Tensor:
    #     return x + delta_t * self._velocity(x, t, condition)

    # def _heun_step(
    #     self,
    #     x: Tensor,
    #     t: Tensor,
    #     condition: Tensor | None,
    #     delta_t: Tensor,
    # ) -> Tensor:
    #     k1 = self._velocity(x, t, condition)
    #     x_pred = x + delta_t * k1
    #     k2 = self._velocity(x_pred, t + delta_t, condition)
    #     return x + 0.5 * delta_t * (k1 + k2)

    # def _rk4_step(
    #     self,
    #     x: Tensor,
    #     t: Tensor,
    #     condition: Tensor | None,
    #     delta_t: Tensor,
    # ) -> Tensor:
    #     half_dt = 0.5 * delta_t
    #     k1 = self._velocity(x, t, condition)
    #     k2 = self._velocity(x + half_dt * k1, t + half_dt, condition)
    #     k3 = self._velocity(x + half_dt * k2, t + half_dt, condition)
    #     k4 = self._velocity(x + delta_t * k3, t + delta_t, condition)
    #     return x + (delta_t / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


INTEGRATOR_STEP_FNS = {
    "Euler": euler_step,
    "RK4": rk4_step,
    "Heun": heun_step,
}
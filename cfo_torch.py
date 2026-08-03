from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn


class ContinuousFlowOperator:
    """Algorithm-only CFO definition for PyTorch.

    This class contains only CFO math/algorithm logic and inference calls.
    State creation, optimization, training loops, and checkpointing are handled outside.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        input_shape: Sequence[int],
        gamma: float = 1e-5,
        spline_type: str = "quintic",
        use_condition: bool = True,
        condition_shape: Sequence[int] | None = None,
    ):
        self.model = model
        self.input_shape = tuple(input_shape)
        self.gamma = float(gamma)
        self.use_condition = bool(use_condition)
        self.condition_shape = tuple(condition_shape) if condition_shape is not None else None
        self.spline_type = str(spline_type)
        if self.spline_type not in {"linear", "quintic"}:
            raise ValueError("`spline_type` must be one of {'linear', 'quintic'}.")
        if self.use_condition and self.condition_shape is None:
            raise ValueError("`condition_shape` must be provided when `use_condition=True`.")

    @staticmethod
    def _reshape_time_like(x: Tensor, spline_coef: Tensor) -> Tensor:
        return x.reshape((x.shape[0],) + (1,) * (spline_coef.ndim - 2))

    def _model_apply(self, x: Tensor, t: Tensor, condition: Tensor | None = None) -> Tensor:
        return self.model(x, t, condition)

    def sample_conditional_path(
        self,
        tau: Tensor,
        spline_coef: Tensor,
        eps: Tensor,
        dt: Tensor | None = None,
    ) -> Tensor:
        if dt is None:
            raise ValueError("`dt` must be provided to sample the conditional path.")

        if self.spline_type == "linear":
            mu_t = spline_coef[:, 0] + tau * spline_coef[:, 1]
        else:
            mu_t = (
                spline_coef[:, 0]
                + tau * spline_coef[:, 1]
                + tau**2 * spline_coef[:, 2]
                + tau**3 * spline_coef[:, 3]
                + tau**4 * spline_coef[:, 4]
                + tau**5 * spline_coef[:, 5]
            )

        gamma_t = self.gamma * (tau**3) * ((1.0 - tau) ** 3)
        return mu_t + gamma_t * eps

    def compute_targets(
        self,
        spline_coef: Tensor,
        tau: Tensor,
        dt: Tensor,
        eps: Tensor,
    ) -> Tensor:
        gamma_prime = self.gamma * (3.0 / dt) * (
            tau**2 * (1.0 - tau) ** 2 * (1.0 - 2.0 * tau)
        )
        if self.spline_type == "linear":
            return (1.0 / dt) * spline_coef[:, 1] + gamma_prime * eps

        return (1.0 / dt) * (
            spline_coef[:, 1]
            + 2.0 * spline_coef[:, 2] * tau
            + 3.0 * spline_coef[:, 3] * tau**2
            + 4.0 * spline_coef[:, 4] * tau**3
            + 5.0 * spline_coef[:, 5] * tau**4
        ) + gamma_prime * eps

    def loss_fn(self, batch) -> Tensor:
        if self.use_condition:
            spline_coef, condition, t_start, t_end, delta_t, eps = batch
        else:
            spline_coef, t_start, t_end, delta_t, eps = batch
            condition = None

        dt = t_end - t_start
        tau = delta_t / dt
        tau = self._reshape_time_like(tau, spline_coef)
        dt = self._reshape_time_like(dt, spline_coef)

        x = self.sample_conditional_path(tau, spline_coef, eps, dt)
        outputs = self._model_apply(x, t_start + delta_t, condition)
        targets = self.compute_targets(spline_coef, tau, dt, eps)
        return torch.mean((outputs - targets) ** 2)

    def _velocity(self, x: Tensor, t: Tensor, condition: Tensor | None = None) -> Tensor:
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, device=x.device, dtype=x.dtype)
        if t.ndim == 0:
            t = torch.full((x.shape[0],), t.item(), device=x.device, dtype=x.dtype)
        return self._model_apply(x, t, condition)

    def _euler_step(
        self,
        x: Tensor,
        t: Tensor,
        condition: Tensor | None,
        delta_t: Tensor,
    ) -> Tensor:
        return x + delta_t * self._velocity(x, t, condition)

    def _heun_step(
        self,
        x: Tensor,
        t: Tensor,
        condition: Tensor | None,
        delta_t: Tensor,
    ) -> Tensor:
        k1 = self._velocity(x, t, condition)
        x_pred = x + delta_t * k1
        k2 = self._velocity(x_pred, t + delta_t, condition)
        return x + 0.5 * delta_t * (k1 + k2)

    def _rk4_step(
        self,
        x: Tensor,
        t: Tensor,
        condition: Tensor | None,
        delta_t: Tensor,
    ) -> Tensor:
        half_dt = 0.5 * delta_t
        k1 = self._velocity(x, t, condition)
        k2 = self._velocity(x + half_dt * k1, t + half_dt, condition)
        k3 = self._velocity(x + half_dt * k2, t + half_dt, condition)
        k4 = self._velocity(x + delta_t * k3, t + delta_t, condition)
        return x + (delta_t / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def infer_at(
        self,
        x_at_s: Tensor,
        s: float,
        t: float,
        steps: int = 50,
        condition: Tensor | None = None,
        method: str = "RK4",
    ) -> Tensor:
        step_fns = {
            "Euler": self._euler_step,
            "Heun": self._heun_step,
            "RK4": self._rk4_step,
        }
        if method not in step_fns:
            raise ValueError("method must be 'Euler', 'Heun', or 'RK4'.")

        step_fn = step_fns[method]
        steps = max(int(steps), 2)
        delta_t = torch.as_tensor((t - s) / (steps - 1), device=x_at_s.device, dtype=x_at_s.dtype)
        t_values = torch.linspace(s, t, steps, device=x_at_s.device, dtype=x_at_s.dtype)

        x = x_at_s
        for i in range(steps - 1):
            t_batch = torch.full((x.shape[0],), t_values[i].item(), device=x.device, dtype=x.dtype)
            x = step_fn(x, t_batch, condition, delta_t)
        return x

    @torch.no_grad()
    def uniform_inference(
        self,
        x_0: Tensor,
        trajectory_points_num: int,
        steps_per_segment: int = 3,
        condition: Tensor | None = None,
        method: str = "RK4",
    ) -> np.ndarray:
        was_training = self.model.training
        self.model.eval()
        try:
            points_num = int(trajectory_points_num)
            segment_steps = max(int(steps_per_segment), 1)
            segment_dt = 1.0 / max(points_num - 1, 1)

            preds = [x_0.detach().cpu().numpy()]
            x = x_0
            for idx in range(points_num - 1):
                s = idx * segment_dt
                t = (idx + 1) * segment_dt

                if condition is None:
                    condition_i = None
                elif condition.ndim == 5:
                    condition_i = condition[:, idx]
                else:
                    condition_i = condition

                x = self.infer_at(
                    x,
                    s=s,
                    t=t,
                    steps=segment_steps + 1,
                    condition=condition_i,
                    method=method,
                )
                preds.append(x.detach().cpu().numpy())

            pred = np.stack(preds, axis=0)
            return np.swapaxes(pred, 0, 1)
        finally:
            self.model.train(was_training)

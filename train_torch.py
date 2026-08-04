"""Project-level train entry helpers (PyTorch version)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple
import copy

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import trange

from utils.data_torch import prepare_torch_data  # analogue of prepare_tf_data, no device-sharding needed
from utils.checkpoints import save_train_state
from utils.logging import ExperimentLogger
from utils.metrics import relative_L2_error, relative_frobenius_error, rmse


@dataclass
class CFOTrainArgs:
    num_epochs: int = 1000
    random_seed: int = 0
    use_wandb: bool = False
    log_mode: str = "auto"
    train_log_interval: int = 1
    console_logging: bool = True
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    do_eval: bool = False
    eval_interval: int = 500
    irregular_time: bool = False
    running_ckpt_dir: str | None = None
    running_ckpt_prefix: str = "running_"
    running_ckpt_interval: int = 0
    running_ckpt_max_to_keep: int = 3
    best_ckpt_dir: str | None = None
    best_ckpt_prefix: str = "best_"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"



@dataclass
class TrainState:
    """Minimal stand-in for flax.training.train_state.TrainState."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    step: int = 0

    def apply_gradients(self) -> "TrainState":
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        return self

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
        }


def _prepare_cfo_train_batch(method, raw_batch, generator: torch.Generator, device: str):
    # NOTE: unlike the JAX version, there's no leading device-shard dimension here
    # (that came from pmap-style prefetch_to_device), so we don't index [0].
    if method.use_condition:
        spline_coef, t_start, t_end, condition, *_ = raw_batch
    else:
        spline_coef, t_start, t_end, *_ = raw_batch
        condition = None

    spline_coef = spline_coef.to(device)
    t_start = t_start.to(device)
    t_end = t_end.to(device)
    dt = t_end - t_start
    delta_t = torch.rand(dt.shape, generator=generator, device=device) * dt

    x0 = spline_coef[:, 0]
    eps = torch.randn(x0.shape, generator=generator, device=device)

    if method.use_condition:
        condition = condition.to(device)
        return (spline_coef, condition, t_start, t_end, delta_t, eps)
    return (spline_coef, t_start, t_end, delta_t, eps)


def _run_cfo_eval(method, state: TrainState, epoch: int, eval_dataset, logger: ExperimentLogger, device: str):
    if method.use_condition:
        x0_eval, target_eval, condition_eval = eval_dataset
        condition_eval = torch.as_tensor(condition_eval, dtype=torch.float32, device=device)
    else:
        x0_eval, target_eval = eval_dataset
        condition_eval = None

    x0_eval = torch.as_tensor(x0_eval, dtype=torch.float32, device=device)
    target_eval = torch.as_tensor(target_eval, dtype=torch.float32, device=device)

    state.model.eval()
    with torch.no_grad():
        pred_eval = method.uniform_inference(
            x0_eval,
            trajectory_points_num=target_eval.shape[1],
            steps_per_segment=2,
            condition=condition_eval,
            method="RK4",
        )
    state.model.train()

    target_np = target_eval.detach().cpu().numpy()
    rel_fro_value = relative_frobenius_error(target_np, pred_eval)
    rmse_value = rmse(target_np, pred_eval)
    rel_l2_value = relative_L2_error(target_np, pred_eval)

    logger.log(
        {
            "eval/epoch": epoch,
            "eval/Relative_L2_Error": rel_l2_value,
            "eval/RMSE": rmse_value,
            "eval/Relative_Frobenius_Error": rel_fro_value,
        },
        commit=False,
    )
    return float(rel_l2_value), float(rmse_value), float(rel_fro_value)


def init_cfo_train_state(
    method,
    *,
    seed: int,
    learning_rate: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.99,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> TrainState:
    torch.manual_seed(seed)
    model = method.model.to(device)

    # Dummy forward pass: not strictly needed for eagerly-shaped nn.Module params
    # (unlike Flax's lazy .init), but kept to sanity-check shapes and to trigger
    # any lazy layers (e.g. nn.LazyLinear) before the optimizer is built.
    x = torch.ones((1,) + tuple(method.input_shape), dtype=torch.float32, device=device)
    t = torch.ones((1,), dtype=torch.float32, device=device)
    with torch.no_grad():
        if method.use_condition:
            c = torch.ones((1,) + tuple(method.condition_shape), dtype=torch.float32, device=device)
            model(x, t, c)
        else:
            model(x, t)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(learning_rate), betas=(float(beta1), float(beta2))
    )
    return TrainState(model=model, optimizer=optimizer)


def train_cfo(method, spline_dataloader, args: CFOTrainArgs, eval_dataset: Optional[Tuple[np.ndarray, np.ndarray]] = None):
    device = args.device
    state = init_cfo_train_state(
        method,
        seed=args.random_seed,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        device=device,
    )
    loss_fn = method.loss_fn  # expected signature: loss_fn(model, batch) -> torch.Tensor

    def cfo_train_step(state: TrainState, batch):
        state.optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(batch)
        loss.backward()
        state.apply_gradients()
        return loss.detach(), state

    logger = ExperimentLogger.create(
        use_wandb=args.use_wandb,
        log_mode=args.log_mode,
        train_log_interval=args.train_log_interval,
        console=args.console_logging,
    )
    num_params = sum(p.numel() for p in state.model.parameters())
    logger.info(f"Model parameters: {int(num_params)}")

    generator = torch.Generator(device=device)
    generator.manual_seed(args.random_seed)

    pbar = trange(args.num_epochs, desc="Training")
    data = map(prepare_torch_data, spline_dataloader)
    data_iter = iter(data)

    loss_log: list[float] = []
    best_state = copy.deepcopy(state.model.state_dict())
    best_l2_error = float("inf")
    best_epoch = -1

    for epoch in pbar:
        batch = _prepare_cfo_train_batch(method, next(data_iter), generator, device)
        loss, state = cfo_train_step(state, batch)

        should_save_running_interval = (
            args.running_ckpt_dir is not None
            and args.running_ckpt_interval > 0
            and (epoch + 1) % args.running_ckpt_interval == 0
        )

        if args.do_eval and (epoch % args.eval_interval == 0) and epoch > 0 and eval_dataset is not None:
            rel_l2_value, rmse_value, rel_fro_value = _run_cfo_eval(method, state, epoch, eval_dataset, logger, device)
            should_save_running_interval = should_save_running_interval or (args.running_ckpt_dir is not None)
            if rel_l2_value < best_l2_error:
                best_l2_error = rel_l2_value
                best_state = copy.deepcopy(state.model.state_dict())
                best_epoch = epoch
                if args.best_ckpt_dir is not None:
                    save_train_state(
                        state,
                        args.best_ckpt_dir,
                        prefix=args.best_ckpt_prefix,
                        step=epoch,
                        max_to_keep=1,
                    )
                logger.info(
                    f"[eval] epoch={epoch} rel_l2={rel_l2_value:.6f} rmse={rmse_value:.6f} rel_fro={rel_fro_value:.6f} [BEST]"
                )
            else:
                logger.info(
                    f"[eval] epoch={epoch} rel_l2={rel_l2_value:.6f} rmse={rmse_value:.6f} rel_fro={rel_fro_value:.6f}"
                )

        if should_save_running_interval:
            save_train_state(
                state,
                args.running_ckpt_dir,
                prefix=args.running_ckpt_prefix,
                step=epoch,
                max_to_keep=args.running_ckpt_max_to_keep,
            )

        loss_value = float(loss)
        loss_log.append(loss_value)
        pbar.set_postfix({"loss": loss_value})
    print("Training complete.")

    return {
        "state": state,
        "best_state": best_state,
        "best_l2_error": best_l2_error,
        "best_epoch": best_epoch,
        "loss_log": loss_log,
    }





__all__ = ["train_cfo", "CFOTrainArgs", "TrainState"]
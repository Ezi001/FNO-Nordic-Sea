"""Train the conditioned CFO model from precomputed Nordic spline stores."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import os

import numpy as np

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cfo_torch import ContinuousFlowOperator
from models.factory import build_model
from train_torch import CFOTrainArgs, train_cfo
from utils.data_torch import ZarrSplineDataset
from utils.metrics import relative_L2_error, relative_frobenius_error, rmse
from utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # run setup
    parser.add_argument("--steps", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)

    # optimisation / CFO dynamics
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--gamma", type=float, default=1e-5)

    # data / model
    parser.add_argument("--dataset", type=str, default="nordic", choices=["nordic"])
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--model", choices=("FNO2d",), default="FNO2d")
    parser.add_argument("--spline-type", choices=("linear", "quintic"), default="quintic")
    
    # parser.add_argument("--train-spline-path", default="data/nordic/splines_train.zarr")
    # parser.add_argument("--train-forcing-path", default="data/nordic/forcing_train.npy")
    # parser.add_argument("--heldout-spline-path", default="data/nordic/splines_heldout.zarr")
    # parser.add_argument("--heldout-forcing-path", default="data/nordic/forcing_heldout.npy")
    
    
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--ckpt-prefix", default="cfo_nordic")
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--running-ckpt-interval", type=int, default=None, help="Save running checkpoint every N epochs; defaults to eval-interval")
    parser.add_argument("--running-ckpt-max-to-keep", type=int, default=3, help="Number of running checkpoints to keep")

    # logging / utility
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    use_condition = True
    effective_lr = float(args.lr)
    effective_beta1 = float(args.beta1)
    effective_beta2 = float(args.beta2)

    if args.dry_run:
        print(args)
        return


    # Loading the train and test data
    train_dataset = ZarrSplineDataset(
        args.train_spline_path,
        forcing_path=args.train_forcing_path,
    )
    heldout_dataset = ZarrSplineDataset(
        args.heldout_spline_path,
        forcing_path=args.heldout_forcing_path,
    )

    if train_dataset.forcing.shape[1:] != heldout_dataset.forcing.shape[1:]:
        raise ValueError(
            "Train and heldout forcing shapes differ: "
            f"{train_dataset.forcing.shape[1:]} vs "
            f"{heldout_dataset.forcing.shape[1:]}"
        )

    spatial_shape = tuple(train_dataset.spline_coef.shape[-2:])
    input_shape = spatial_shape + (1,)
    condition_shape = tuple(train_dataset.forcing.shape[1:])

    # 2. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    heldout_loader = DataLoader(
        heldout_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    model = build_model(args.model, input_shape=input_shape, use_condition=True, condition_shape=condition_shape)

    print("=== CFO Training Run ===")
    print(f"dataset={args.dataset} model={args.model} spline_type={args.spline_type}")
    print(f"seed={args.seed} epochs={args.epochs} batch_size={args.batch_size} spline_batch_size={args.spline_batch_size}")
    print(f"lr={effective_lr} beta1={effective_beta1} beta2={effective_beta2}")
    print(f"eval_interval={args.eval_interval} do_eval={not args.no_eval}")
    print(f"train/eval/test sizes = {len(train_data)}/{len(eval_data)}/{len(test_data)}")
    print(f"input_shape={input_shape}")
    running_ckpt_interval = args.eval_interval if args.running_ckpt_interval is None else args.running_ckpt_interval
    print(f"checkpoint_policy running_every={running_ckpt_interval} keep={args.running_ckpt_max_to_keep} best_keep=1")


    print("starting training...")


    method = ContinuousFlowOperator(
        model=model,
        input_shape=input_shape,
        gamma=args.gamma,
        spline_type=args.spline_type,
        use_condition=True,
        condition_shape=condition_shape,
    )

    print("=== DATA SUMMARY ===")
    print(f"train trajectories: {train_dataset.n_trajectories}")
    print(f"heldout trajectories: {heldout_dataset.n_trajectories}")
    print(f"intervals per trajectory: {train_dataset.n_intervals}")
    print(f"spline coefficient shape: {tuple(train_dataset.spline_coef.shape)}")
    print(f"forcing shape: {tuple(train_dataset.forcing.shape)}")
    print(f"model input shape: {input_shape}")
    print(f"condition shape: {condition_shape}")

    ckpt_dir = Path(args.ckpt_dir).resolve()

    train_args = CFOTrainArgs(
        num_steps=args.steps,
        random_seed=args.seed,
        use_wandb=args.use_wandb,
        learning_rate=effective_lr,
        beta1=effective_beta1,
        beta2=effective_beta2,
        do_eval=not args.no_eval,
        eval_interval=args.eval_interval,
        running_ckpt_dir=str(ckpt_dir / "running"),
        running_ckpt_prefix=args.ckpt_prefix,
        running_ckpt_interval=running_ckpt_interval,
        running_ckpt_max_to_keep=args.running_ckpt_max_to_keep,
        best_ckpt_dir=str(ckpt_dir / "best"),
        best_ckpt_prefix=args.ckpt_prefix,
    )

    eval_dataset = (eval_data[:, 0], eval_data)

    print("Starting training loop...")
    train_output = train_cfo(method, spline_loader, train_args, eval_dataset=eval_dataset)

    state_for_test = train_output["state"]
    if np.isfinite(train_output["best_l2_error"]):
        state_for_test = train_output["best_state"]

    print("Running final test inference...")
    test_pred = method.uniform_inference(
        state_for_test,
        test_data[:, 0],
        trajectory_points_num=test_data.shape[1],
        steps_per_segment=2,
        method="RK4",
    )
    # calculating metrics
    test_rmse = rmse(test_data, test_pred)
    test_rel_l2 = relative_L2_error(test_data, test_pred)
    test_rel_fro = relative_frobenius_error(test_data, test_pred)

    best_epoch = train_output["best_epoch"]
    best_rel_l2 = train_output["best_l2_error"]

    if np.isfinite(best_rel_l2):
        print(f"Best eval checkpoint: epoch={best_epoch} rel_l2={best_rel_l2:.6f}")
    else:
        print("Best eval checkpoint: none (evaluation disabled or not run)")

    print("Final held-out test metrics:")
    print(f"  Relative L2 Error: {float(test_rel_l2):.6f}")
    print(f"  RMSE: {float(test_rmse):.6f}")
    print(f"  Relative Frobenius Error: {float(test_rel_fro):.6f}")

    ckpt_dir = Path(args.ckpt_dir).resolve()
    print(f"Running checkpoints: dir={ckpt_dir / 'running'} keep={args.running_ckpt_max_to_keep}")
    print(f"Best checkpoint: dir={ckpt_dir / 'best'} keep=1")



if __name__ == "__main__":
    main()

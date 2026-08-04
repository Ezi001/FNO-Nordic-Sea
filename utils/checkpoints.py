from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch


def _manager_directory(ckpt_dir: str, prefix: str) -> Path:
    """Return checkpoint directory for a run prefix."""
    return Path(ckpt_dir) / prefix



def _checkpoint_path(ckpt_dir: str, prefix: str, step: int) -> Path:
    """Checkpoint file path for a given step."""
    return _manager_directory(ckpt_dir, prefix) / f"checkpoint_{step}.pt"



def _cleanup_old_checkpoints(
    ckpt_dir: str,
    prefix: str,
    max_to_keep: int,
) -> None:
    """Keep only the newest max_to_keep checkpoints."""
    ckpt_dir_path = _manager_directory(ckpt_dir, prefix)

    checkpoints = sorted(
        ckpt_dir_path.glob("checkpoint_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )

    excess = len(checkpoints) - max_to_keep
    if excess > 0:
        for ckpt in checkpoints[:excess]:
            ckpt.unlink(missing_ok=True)



def save_train_state(
    state: Any,
    ckpt_dir: str,
    prefix: str,
    step: int,
    max_to_keep: int = 5,
) -> None:
    """Save training state at a given step."""
    step = int(step)

    save_dir = _manager_directory(ckpt_dir, prefix)
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = _checkpoint_path(ckpt_dir, prefix, step)

    torch.save(state, ckpt_path)

    _cleanup_old_checkpoints(
        ckpt_dir,
        prefix,
        max_to_keep=max_to_keep,
    )



def load_train_state(
    ckpt_dir: str,
    prefix: str,
    step: Optional[int] = None,
    map_location: str | torch.device = "cpu",
) -> Any:
    """Restore training state from a checkpoint step or latest."""
    ckpt_dir_path = _manager_directory(ckpt_dir, prefix)

    if not ckpt_dir_path.exists():
        raise FileNotFoundError(
            f"No checkpoints found in '{ckpt_dir_path}'."
        )

    if step is None:
        checkpoints = sorted(
            ckpt_dir_path.glob("checkpoint_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )

        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoints found in '{ckpt_dir_path}'."
            )

        ckpt_path = checkpoints[-1]
    else:
        ckpt_path = _checkpoint_path(ckpt_dir, prefix, int(step))

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint step {step} not found: '{ckpt_path}'."
            )

    return torch.load(
        ckpt_path,
        map_location=map_location,
        weights_only=False,
    )

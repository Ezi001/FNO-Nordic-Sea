"""Merged data helpers (loaders/datasets/splines) - PyTorch version."""

from __future__ import annotations

import itertools
from typing import Iterator

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, TensorDataset

from .splines_torch import linear_spline, quintic_spline_batch

# NOTE: `linear_spline` / `quintic_spline_batch` are imported unchanged here.
# If those are implemented in JAX (jnp), they'll need their own torch port —
# happy to do that one too if you share splines.py.


def prepare_torch_data(batch):
    """Analogue of `prepare_tf_data`.

    The JAX version reshaped each array to `(num_devices, -1, ...)` so it could
    be sharded across devices for `pmap`. That reshape has no equivalent need
    here — a single-process PyTorch `DataLoader` batch is already just a
    (batch, ...) tensor (or tuple of tensors). This is effectively a no-op,
    kept so `train.py`'s `map(prepare_torch_data, dataloader)` still works and
    so dtype coercion lives in one place if you need it later.
    """
    if isinstance(batch, (tuple, list)):
        return tuple(_to_tensor(x) for x in batch)
    return _to_tensor(batch)


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(np.asarray(x))


class CudaPrefetcher:
    """Analogue of `flax.jax_utils.prefetch_to_device`.

    Overlaps host->device transfer of the next batch with compute on the
    current batch, using a side CUDA stream. On CPU this just falls back to a
    plain iterator (there's nothing to prefetch onto).

    Usage:
        loader = CudaPrefetcher(dataloader, device="cuda")
        for batch in loader:
            ...
    """

    def __init__(self, loader, device: str = "cuda", size: int = 2):
        self.loader = loader
        self.device = torch.device(device)
        self.size = size  # kept for API parity with prefetch_to_device(iterator, size); unused beyond depth=1

    def __iter__(self):
        if self.device.type != "cuda":
            # Nothing to prefetch onto — just move tensors to device as we go.
            for batch in self.loader:
                yield tree_to_device(batch, self.device)
            return

        stream = torch.cuda.Stream()
        it = iter(self.loader)

        def _preload():
            try:
                nxt = next(it)
            except StopIteration:
                return None
            with torch.cuda.stream(stream):
                nxt = tree_to_device(nxt, self.device, non_blocking=True)
            return nxt

        next_batch = _preload()
        while next_batch is not None:
            torch.cuda.current_stream().wait_stream(stream)
            batch = next_batch
            next_batch = _preload()
            yield batch


def tree_to_device(xs, device, non_blocking: bool = False):
    if isinstance(xs, torch.Tensor):
        return xs.to(device, non_blocking=non_blocking)
    if isinstance(xs, (tuple, list)):
        return type(xs)(tree_to_device(x, device, non_blocking) for x in xs)
    if isinstance(xs, dict):
        return {k: tree_to_device(v, device, non_blocking) for k, v in xs.items()}
    return xs


class _RepeatingLoader:
    """Wraps a DataLoader so iterating it yields `num_epochs` worth of batches
    back-to-back, reshuffling each pass — the torch analogue of
    `tf.data.Dataset.repeat(num_epochs)` placed *before* `.batch(...)`.
    """

    def __init__(self, dataloader: DataLoader, num_epochs: int):
        self.dataloader = dataloader
        self.num_epochs = num_epochs

    def __iter__(self) -> Iterator:
        # Each call to iter(self.dataloader) reshuffles (shuffle=True), same
        # as tf.data's per-epoch reshuffling behavior.
        return itertools.chain.from_iterable(
            iter(self.dataloader) for _ in range(self.num_epochs)
        )

    def __len__(self) -> int:
        return len(self.dataloader) * self.num_epochs


def build_dataloader(X, t1=None, t2=None, c=None, y=None, batch_size=64, num_epochs=1, seed=0):
    """Create a DataLoader yielding `(X, [t1], [t2], [c])` batches.

    Args:
        X: Primary input tensor.
        t1: First optional auxiliary tensor (e.g., start time).
        t2: Second optional auxiliary tensor (e.g., end time).
        c: Optional condition tensor.
        y: Backward-compatible alias for `t1`.
    """
    if t1 is None and y is not None:
        t1 = y

    tensors = (X,) + ((t1,) if t1 is not None else ()) + ((t2,) if t2 is not None else ()) + ((c,) if c is not None else ())
    tensors = tuple(_to_tensor(t).float() if np.issubdtype(np.asarray(t).dtype, np.floating) else _to_tensor(t) for t in tensors)

    dataset = TensorDataset(*tensors)

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        num_workers=0,  # bump if you need background workers; dataset here is in-memory
        pin_memory=torch.cuda.is_available(),
    )
    return _RepeatingLoader(loader, num_epochs=num_epochs)




def load_partial_data(data, ratio, seed=43):
    """Randomly subsample a fraction of time snapshots from each trajectory.

    Returns:
        partial_data: np.ndarray of shape (B, K, ...)
        time_values: np.ndarray of shape (B, K), normalized to [0, 1]
    """
    data = np.asarray(data)
    if data.ndim < 2:
        raise ValueError("`data` must have shape (B, T, ...).")

    if not (0.0 < float(ratio) <= 1.0):
        raise ValueError(f"`ratio` must be in (0, 1], got {ratio}.")

    B, T = data.shape[:2]
    K = max(2, int(np.round(T * float(ratio))))
    K = min(K, T)

    rng = np.random.default_rng(seed)
    rand = rng.random((B, T))
    perm = np.argsort(rand, axis=1)
    time_indices = np.sort(perm[:, :K], axis=1)

    batch_idx = np.arange(B)[:, None]
    partial_data = data[batch_idx, time_indices]
    denom = max(T - 1, 1)
    time_values = time_indices.astype(np.float32) / float(denom)
    return partial_data, time_values


def select_data_split(splits: dict[str, np.ndarray], split: str):
    """Select one split from a split dictionary (e.g., `{"train", "eval", "test"}`)."""
    split_key = str(split).lower()
    if split_key not in splits:
        raise ValueError(f"Unsupported split='{split}'. Expected one of {tuple(splits.keys())}.")
    return splits[split_key]


__all__ = [
    "prepare_torch_data",
    "CudaPrefetcher",
    "tree_to_device",
    "build_dataloader",
    "load_partial_data",
    "select_data_split",
    "linear_spline",
    "quintic_spline_batch",
]
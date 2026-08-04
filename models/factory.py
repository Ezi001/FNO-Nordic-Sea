"""Model factory for project-level imports."""

from __future__ import annotations

from typing import Sequence

from models.fno_torch import FNO2d


def build_model(
    model_name: str,
    input_shape: Sequence[int],
    use_condition: bool = True,
    condition_shape: Sequence[int] | None = None,
):
    """Instantiate a model from a standardized model name."""
    name_l = model_name.strip().lower()

    in_channels = input_shape[-1] if len(input_shape) >= 1 else 1
    out_channels = in_channels

    if condition_shape is None:
        condition_channels = in_channels
    else:
        condition_channels = condition_shape[-1]

    if name_l == "fno2d":
        return FNO2d(
            num_channels=out_channels,
            in_channels=in_channels,
            condition_channels=condition_channels,
            use_condition=use_condition,
        )

    raise ValueError(
        f"Unsupported model_name='{model_name}'. "
        "Supported: FNO2d"
    )


__all__ = ["build_model"]
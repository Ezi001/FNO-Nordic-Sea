"""Model factory for project-level imports."""

from __future__ import annotations

from typing import Sequence

#from models.dit import DiT
from models.fno_torch import FNO2d
#from models.mlp import SimpleMLP
#from models.unet import UNet1D, UNet2D


def build_model(
	model_name: str,
	input_shape: Sequence[int],
	use_condition: bool = True,
):
	"""Instantiate a model from a standardized model name."""
	name = model_name.strip()
	name_l = name.lower()

	out_channels = input_shape[-1] if len(input_shape) >= 2 else 1

	if name_l == "fno2d":
		return FNO2d(num_channels=out_channels, use_condition=use_condition)

	raise ValueError(
		f"Unsupported model_name='{model_name}'. "
		"Supported: FNO1d, FNO2d"
	)

__all__ = ["build_model"]
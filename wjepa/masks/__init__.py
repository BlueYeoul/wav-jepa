"""Mask generation utilities for W-JEPA."""

from .collator import DynamicMaskCollator1D as MaskCollator
from .distance import compute_mask_distance

__all__ = ["MaskCollator", "compute_mask_distance"]

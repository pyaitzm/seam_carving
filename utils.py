"""
Utilities and configuration for the seam-carving project.

This module defines:
  - Config: Immutable dataclass storing configuration for energy mode,
    downsizing, and mask constants.
  - Lightweight helpers for resizing, rotation, and image saving.
  - prepare_mask: Validation and one-step binarization of masks.

Design notes:
  - I/O uses uint8 (OpenCV). Core computation uses float64.
  - Masks are validated and converted once to bool (True = active).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Config:
    """Immutable configuration container for the seam-carving algorithm."""
    use_forward_energy: bool = True
    should_downsize: bool = True
    downsize_width: int = 500
    energy_mask_const: float = 100000.0
    mask_threshold: int = 10


def resize(image: np.ndarray, width: int) -> np.ndarray:
    """Resize an image to a target width, preserving aspect ratio."""
    h, w = image.shape[:2]
    dim = (width, int(h * width / float(w)))
    return cv2.resize(image, dim)


def rotate_image(image: np.ndarray, clockwise: bool) -> np.ndarray:
    """Rotate an image 90 degrees clockwise or counterclockwise."""
    k = 1 if clockwise else 3
    return np.rot90(image, k)


def save_uint8(path: str, img: np.ndarray) -> None:
    """Save an image to disk as uint8, clipping to [0, 255] if needed."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    ok = cv2.imwrite(path, img)
    if not ok:
        raise RuntimeError(f"Failed to write image to: {path}")


def prepare_mask(
    mask_img: Optional[np.ndarray],
    shape_hw: Tuple[int, int],
    threshold: int,
    name: str = "mask",
    allow_none: bool = True,
) -> Optional[np.ndarray]:
    """Validate and binarize a mask image into a boolean array.

    Ensures the mask matches the working image size, converts to grayscale
    if needed, and thresholds it once to produce a bool array.

    Returns:
        Boolean mask array of shape (H, W) or None if no mask is provided.
    """
    if mask_img is None:
        if allow_none:
            return None
        raise ValueError(f"{name} is required but was None")

    if mask_img.ndim == 3:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

    h, w = shape_hw
    mh, mw = mask_img.shape[:2]
    if (mh, mw) != (h, w):
        raise ValueError(
            f"{name} size mismatch: expected {(h, w)}, got {(mh, mw)}. "
            "Ensure masks are matched after any downsizing."
        )

    if mask_img.dtype != np.uint8:
        mask_img = mask_img.astype(np.int32)
    mask_bool = mask_img > int(threshold)
    return mask_bool

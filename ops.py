from __future__ import annotations
from typing import Optional, Tuple, Callable

import numpy as np

from utils import Config, rotate_image
from seams import (
    get_minimum_seam,
    remove_seam, remove_seam_grayscale,
    add_seam, add_seam_grayscale
)

# Type alias: on_seam(image_float64, seam_idx_int64) -> None
OnSeam = Optional[Callable[[np.ndarray, np.ndarray], None]]


def seams_removal(
    im: np.ndarray,
    num_remove: int,
    cfg: Config,
    mask: Optional[np.ndarray] = None,
    on_seam: OnSeam = None
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Remove `num_remove` vertical seams; optionally call `on_seam` before each removal."""
    for _ in range(int(num_remove)):
        seam_idx, boolmask = get_minimum_seam(im, cfg, mask)
        if on_seam is not None:
            on_seam(im, seam_idx)  # visualize current seam on current image
        im = remove_seam(im, boolmask)
        if mask is not None:
            mask = remove_seam_grayscale(mask, boolmask)
    return im, mask


def seams_insertion(
    im: np.ndarray,
    num_add: int,
    cfg: Config,
    mask: Optional[np.ndarray] = None
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Insert `num_add` vertical seams.
    Note: we do not visualize insertion seams by default to keep runtime and memory low.
    """
    seams_record = []
    temp_im = im.copy()
    temp_mask = mask.copy() if mask is not None else None

    for _ in range(int(num_add)):
        seam_idx, boolmask = get_minimum_seam(temp_im, cfg, temp_mask)
        seams_record.append(seam_idx)
        temp_im = remove_seam(temp_im, boolmask)
        if temp_mask is not None:
            temp_mask = remove_seam_grayscale(temp_mask, boolmask)

    seams_record.reverse()

    for _ in range(int(num_add)):
        seam = seams_record.pop()
        im = add_seam(im, seam)
        if mask is not None:
            mask = add_seam_grayscale(mask, seam)

        # update the remaining seam indices
        for remaining_seam in seams_record:
            remaining_seam[np.where(remaining_seam >= seam)] += 2

    return im, mask


def seam_carve(
    im: np.ndarray,
    dy: int,
    dx: int,
    cfg: Config,
    mask: Optional[np.ndarray] = None,
    on_seam: OnSeam = None
) -> np.ndarray:
    if im.dtype != np.float64:
        im = im.astype(np.float64)

    if mask is not None and mask.dtype != np.bool_:
        mask = mask.astype(np.bool_)

    h, w = im.shape[:2]
    assert h + dy > 0 and w + dx > 0 and dy <= h and dx <= w

    output = im

    if dx < 0:
        output, mask = seams_removal(output, -dx, cfg, mask, on_seam=on_seam)
    elif dx > 0:
        output, mask = seams_insertion(output, dx, cfg, mask)

    if dy < 0:
        output = rotate_image(output, True)
        if mask is not None:
            mask = rotate_image(mask, True)
        output, mask = seams_removal(output, -dy, cfg, mask, on_seam=on_seam)
        output = rotate_image(output, False)
        if mask is not None:
            mask = rotate_image(mask, False)
    elif dy > 0:
        output = rotate_image(output, True)
        if mask is not None:
            mask = rotate_image(mask, True)
        output, mask = seams_insertion(output, dy, cfg, mask)
        output = rotate_image(output, False)
        if mask is not None:
            mask = rotate_image(mask, False)

    return output


def object_removal(
    im: np.ndarray,
    rmask: np.ndarray,
    cfg: Config,
    mask: Optional[np.ndarray] = None,
    horizontal_removal: bool = False,
    on_seam: OnSeam = None
) -> np.ndarray:
    if im.dtype != np.float64:
        im = im.astype(np.float64)

    if rmask.dtype != np.bool_:
        rmask = rmask.astype(np.bool_)
    if mask is not None and mask.dtype != np.bool_:
        mask = mask.astype(np.bool_)

    output = im
    h, w = im.shape[:2]

    if horizontal_removal:
        output = rotate_image(output, True)
        rmask = rotate_image(rmask, True)
        if mask is not None:
            mask = rotate_image(mask, True)

    while np.any(rmask):
        seam_idx, boolmask = get_minimum_seam(output, cfg, mask, rmask)
        if on_seam is not None:
            on_seam(output, seam_idx)
        output = remove_seam(output, boolmask)
        rmask = remove_seam_grayscale(rmask, boolmask)
        if mask is not None:
            mask = remove_seam_grayscale(mask, boolmask)

    num_add = (h if horizontal_removal else w) - output.shape[1]
    output, mask = seams_insertion(output, num_add, cfg, mask)
    if horizontal_removal:
        output = rotate_image(output, False)

    return output

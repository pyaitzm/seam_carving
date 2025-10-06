from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
from numba import njit

from utils import Config
from energy import forward_energy, backward_energy


# ==========================
# Numba-compiled DP helpers
# ==========================
@njit(cache=True)
def _dp_accumulate(M: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Given an energy matrix M (float64 HxW), do in-place DP accumulation (except for M[0])
    and return:
      - backtrack: int64 HxW of predecessor column indices
      - end_j: the column index in the last row where the min accumulated energy ends
    Note: Works purely with NumPy/Numba primitives (no cv2/scipy here).
    """
    h, w = M.shape
    backtrack = np.zeros((h, w), dtype=np.int64)

    for i in range(1, h):
        for j in range(w):
            if j == 0:
                # candidates: (i-1, j), (i-1, j+1)
                left_j = j
                right_j = j + 1
                # choose min of two
                if M[i - 1, left_j] <= M[i - 1, right_j]:
                    backtrack[i, j] = left_j
                    min_energy = M[i - 1, left_j]
                else:
                    backtrack[i, j] = right_j
                    min_energy = M[i - 1, right_j]
            elif j == w - 1:
                # candidates: (i-1, j-1), (i-1, j)
                left_j = j - 1
                right_j = j
                if M[i - 1, left_j] <= M[i - 1, right_j]:
                    backtrack[i, j] = left_j
                    min_energy = M[i - 1, left_j]
                else:
                    backtrack[i, j] = right_j
                    min_energy = M[i - 1, right_j]
            else:
                # candidates: (i-1,j-1), (i-1,j), (i-1,j+1)
                jm1 = j - 1
                jp1 = j + 1
                a = M[i - 1, jm1]
                b = M[i - 1, j]
                c = M[i - 1, jp1]
                # argmin of (a, b, c)
                if a <= b and a <= c:
                    backtrack[i, j] = jm1
                    min_energy = a
                elif b <= a and b <= c:
                    backtrack[i, j] = j
                    min_energy = b
                else:
                    backtrack[i, j] = jp1
                    min_energy = c

            M[i, j] = M[i, j] + min_energy

    # find argmin on the last row
    end_j = 0
    min_val = M[h - 1, 0]
    for j in range(1, w):
        if M[h - 1, j] < min_val:
            min_val = M[h - 1, j]
            end_j = j
    return backtrack, end_j


@njit(cache=True)
def _dp_backtrack(backtrack: np.ndarray, end_j: int) -> np.ndarray:
    """
    Reconstruct the seam indices given a backtrack table and the last-row column end_j.
    Returns seam_idx of shape (H,) int64 such that seam_idx[i] is the column in row i.
    """
    h, w = backtrack.shape
    seam = np.empty(h, dtype=np.int64)
    j = end_j
    for i in range(h - 1, -1, -1):
        seam[i] = j
        j = backtrack[i, j]
    return seam


# ==============
# SEAM HELPERS
# ==============
def add_seam(im: np.ndarray, seam_idx: np.ndarray) -> np.ndarray:
    """Add a vertical seam (color image)."""
    h, w = im.shape[:2]
    output = np.zeros((h, w + 1, 3), dtype=im.dtype)
    for row in range(h):
        col = int(seam_idx[row])
        for ch in range(3):
            if col == 0:
                p = np.average(im[row, col: col + 2, ch])
                output[row, col, ch] = im[row, col, ch]
                output[row, col + 1, ch] = p
                output[row, col + 1:, ch] = im[row, col:, ch]
            else:
                p = np.average(im[row, col - 1: col + 1, ch])
                output[row, : col, ch] = im[row, : col, ch]
                output[row, col, ch] = p
                output[row, col + 1:, ch] = im[row, col:, ch]
    return output


def add_seam_grayscale(im: np.ndarray, seam_idx: np.ndarray) -> np.ndarray:
    """Add a vertical seam (grayscale image)."""
    h, w = im.shape[:2]
    output = np.zeros((h, w + 1), dtype=im.dtype)
    for row in range(h):
        col = int(seam_idx[row])
        if col == 0:
            p = np.average(im[row, col: col + 2])
            output[row, col] = im[row, col]
            output[row, col + 1] = p
            output[row, col + 1:] = im[row, col:]
        else:
            p = np.average(im[row, col - 1: col + 1])
            output[row, : col] = im[row, : col]
            output[row, col] = p
            output[row, col + 1:] = im[row, col:]
    return output


def remove_seam(im: np.ndarray, boolmask: np.ndarray) -> np.ndarray:
    """Remove a vertical seam (color image) given bool mask (False marks the seam)."""
    h, w = im.shape[:2]
    boolmask3c = np.stack((boolmask, boolmask, boolmask), axis=2)
    return im[boolmask3c].reshape((h, w - 1, 3))


def remove_seam_grayscale(im: np.ndarray, boolmask: np.ndarray) -> np.ndarray:
    """Remove a vertical seam (grayscale image) given bool mask (False marks the seam)."""
    h, w = im.shape[:2]
    return im[boolmask].reshape((h, w - 1))


# =======================
# CORE DP / SEAM SEARCH
# =======================
def get_minimum_seam(
    im: np.ndarray,
    cfg: Config,
    mask: Optional[np.ndarray] = None,         # bool or None
    remove_mask: Optional[np.ndarray] = None   # bool or None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the seam of minimum energy.

    Strategy for Numba safety/speed:
      1) Compute energy map M (may use cv2/scipy) OUTSIDE JIT.
      2) Apply masks (numpy) OUTSIDE JIT.
      3) Run JIT only on the pure DP loops (accumulation + backtrack).

    Returns:
        seam_idx: int64 array (H,) with seam column per row
        boolmask: bool HxW where False marks the seam
    """
    h, w = im.shape[:2]
    energyfn = forward_energy if cfg.use_forward_energy else backward_energy

    # Step 1: energy map (float64 HxW)
    M = energyfn(im)

    # Step 2: apply masks (numpy)
    if mask is not None:
        M[mask] = cfg.energy_mask_const
    if remove_mask is not None:
        M[remove_mask] = -cfg.energy_mask_const * 100

    # Step 3: DP accumulation + backtrack (Numba)
    backtrack, end_j = _dp_accumulate(M)
    seam_idx = _dp_backtrack(backtrack, end_j)

    # Build boolmask (numpy, trivial cost compared to DP)
    boolmask = np.ones((h, w), dtype=np.bool_)
    for i in range(h):
        boolmask[i, seam_idx[i]] = False

    return seam_idx, boolmask

"""
Energy functions for seam carving.

This module provides energy map computations used by the dynamic-programming
seam search:
  - backward_energy: gradient-magnitude (backward) energy.
  - forward_energy: forward-looking energy that estimates disruption cost.

Notes:
  - Input images are expected as uint8 for I/O; they may be cast to float64
    internally for computation.
  - The caller chooses which energy to use through Config elsewhere in the
    codebase.
"""

from __future__ import annotations
import numpy as np
import cv2
from scipy import ndimage as ndi


def backward_energy(im: np.ndarray) -> np.ndarray:
    """
    Simple gradient magnitude energy map on color images.
    Input: im (HxWx3), dtype float64 preferred.
    Output: energy map (HxW), float64.
    """
    if im.dtype != np.float64:
        im = im.astype(np.float64)

    xgrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=1, mode='wrap')
    ygrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=0, mode='wrap')
    grad_mag = np.sqrt(np.sum(xgrad**2, axis=2) + np.sum(ygrad**2, axis=2))
    return grad_mag


def forward_energy(im: np.ndarray) -> np.ndarray:
    """
    Forward energy algorithm as described in
    "Improved Seam Carving for Video Retargeting" (Rubinstein, Shamir, Avidan).
    Returns per-pixel cost of removing a vertical seam.
    """
    im_gray = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = im_gray.shape[:2]

    energy = np.zeros((h, w), dtype=np.float64)
    m = np.zeros((h, w), dtype=np.float64)

    U = np.roll(im_gray, 1, axis=0)
    L = np.roll(im_gray, 1, axis=1)
    R = np.roll(im_gray, -1, axis=1)

    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU

    for i in range(1, h):
        mU = m[i - 1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)

        mULR = np.array([mU, mL, mR])     # (3, w)
        cULR = np.array([cU[i], cL[i], cR[i]])

        mULR = mULR + cULR
        argmins = np.argmin(mULR, axis=0)
        m[i] = mULR[argmins, np.arange(w)]
        energy[i] = cULR[argmins, np.arange(w)]

    return energy

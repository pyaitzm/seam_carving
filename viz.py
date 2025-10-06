from __future__ import annotations

import os
from typing import Optional, List

import numpy as np
import cv2

# Prefer imageio for GIF writing, fall back to Pillow if present.
try:
    import imageio.v2 as imageio  # imageio >= 2
except Exception:
    imageio = None

try:
    from PIL import Image
except Exception:
    Image = None


def _pad_to_size_bgr(img_bgr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Pad BGR image to (target_h, target_w) with solid white (255,255,255)."""
    h, w = img_bgr.shape[:2]
    dh = max(0, target_h - h)
    dw = max(0, target_w - w)
    if dh == 0 and dw == 0:
        return img_bgr
    return cv2.copyMakeBorder(img_bgr, 0, dh, 0, dw, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))


class VizGifRecorder:
    """
    GIF recorder for seam carving.

    - on_seam(im_f64, seam_idx): draws seam (red) on a uint8 copy and stores an RGB frame.
    - Frames are automatically normalized to a common size:
        * The recorder maintains a (target_h, target_w). On the first frame, it is set to that frame size.
        * If a later frame is larger (e.g., after a rotate for horizontal removal),
          the target grows and earlier frames are retro-padded.
        * If a later frame is smaller (typical when removing seams), it is padded to the target.
    - close(): writes the animated GIF via imageio (preferred) or Pillow fallback.
    """
    def __init__(self, gif_path: str, every: int = 1, max_frames: Optional[int] = None, fps: int = 12):
        self.gif_path = gif_path
        self.every = max(1, int(every))
        self.max_frames = max_frames if (max_frames is None or max_frames > 0) else None
        self.fps = max(1, int(fps))

        out_dir = os.path.dirname(os.path.abspath(gif_path))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        self._frames: List[np.ndarray] = []   # list of RGB frames (all equal size when writing)
        self._step = 0
        self._target_h: Optional[int] = None
        self._target_w: Optional[int] = None

    def _ensure_target_and_retrofit(self, h: int, w: int) -> None:
        """
        Ensure (target_h, target_w) can accommodate a frame of size (h, w).
        If this frame is larger than the current target, grow the target and retro-pad prior frames.
        """
        if self._target_h is None or self._target_w is None:
            self._target_h, self._target_w = h, w
            return

        grew = False
        if h > self._target_h:
            self._target_h = h
            grew = True
        if w > self._target_w:
            self._target_w = w
            grew = True

        if grew and self._frames:
            # Retro-pad all existing frames to new target
            new_frames: List[np.ndarray] = []
            for fr in self._frames:
                fr_bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
                fr_bgr = _pad_to_size_bgr(fr_bgr, self._target_h, self._target_w)
                new_frames.append(cv2.cvtColor(fr_bgr, cv2.COLOR_BGR2RGB))
            self._frames = new_frames

    def on_seam(self, im_f64: np.ndarray, seam_idx: np.ndarray) -> None:
        """
        Record a frame with the current seam drawn in red.
        im_f64: HxWx3 float64 image (BGR-like)
        seam_idx: (H,) int64 column for each row
        """
        # sampling / frame cap
        if (self._step % self.every) != 0:
            self._step += 1
            return
        if self.max_frames is not None and len(self._frames) >= self.max_frames:
            self._step += 1
            return

        # Make drawable BGR copy
        frame_bgr = np.clip(im_f64, 0, 255).astype(np.uint8).copy()
        h, w = frame_bgr.shape[:2]

        # Grow target if needed (handles rotations or any larger frame)
        self._ensure_target_and_retrofit(h, w)

        # Draw seam in red (BGR=(0,0,255))
        H = frame_bgr.shape[0]
        for i in range(H):
            j = int(seam_idx[i])
            if 0 <= j < frame_bgr.shape[1]:
                frame_bgr[i, j] = (0, 0, 255)

        # Pad to target and convert to RGB for the encoder
        frame_bgr = _pad_to_size_bgr(frame_bgr, self._target_h, self._target_w)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._frames.append(frame_rgb)
        self._step += 1

    def close(self) -> None:
        """Write the animated GIF to disk (if any frames were recorded)."""
        if not self._frames:
            return

        duration_sec = 1.0 / float(self.fps)

        if imageio is not None:
            imageio.mimsave(self.gif_path, self._frames, format="GIF", duration=duration_sec, loop=0)
            return

        if Image is not None:
            pil_frames = [Image.fromarray(f) for f in self._frames]
            pil_frames[0].save(
                self.gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(duration_sec * 1000),
                loop=0,
                disposal=2,
            )
            return

        raise RuntimeError(
            "Cannot write GIF: neither `imageio` nor `Pillow` is installed. "
            "Install one of them, e.g. `pip install imageio`."
        )

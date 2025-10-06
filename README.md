# Seam Carving

**Content-Aware Image Resizing and Object Removal**  
_refactored from [andrewdcampbell/seam-carving](https://github.com/andrewdcampbell/seam-carving)_

---

## Overview

**Seam carving** is a content-aware image resizing technique that removes or inserts low-energy pixel paths (“seams”) to change an image’s dimensions or remove objects while preserving visually important regions.

This refactored version provides:
- **forward and backward energy functions** (Avidan & Shamir, Rubinstein et al.)  
- **intelligent seam insertion/removal** (vertical & horizontal)  
- **protective and removal masks** for region control  
- **numba-accelerated DP** (safe placement)
- **optional GIF visualization** of carved seams (constant-size frames with padding)  
- **auto-orientation** for object removal  
- **deterministic dry-run (`--plan-only`) mode** for debugging  
- **modular, maintainable architecture** with no global state  

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/pyaitzm/seam_carving.git
cd seam_carving
pip install -r requirements.txt
```

optional (for GIF export):
```bash
pip install imageio Pillow
```

---

### 2. Basic Usage

#### resize an image (content-aware)
```bash
python seam_carving.py \
  --mode resize \
  --image demo/in.jpg \
  --output demo/out_resize.jpg \
  --dy -40 --dx -80

```

#### remove an object using a mask
```bash
python seam_carving.py \
  --mode remove \
  --image demo/in.jpg \
  --output demo/out_remove.jpg \
  --rmask demo/rmask.png
```

#### visualize seam removal as a GIF
```bash
python seam_carving.py \
  --mode resize \
  --image demo/in.jpg \
  --output demo/out_resize_viz.jpg \
  --dy -40 --dx -80 \
  --viz-gif demo/resize_seams.gif \
  --viz-every 2 \
  --viz-fps 12
```

#### preview the plan (no processing)
```bash
python seam_carving.py \
  --mode resize \
  --image demo/in.jpg \
  --output /tmp/ignore.jpg \
  --dy -10 --dx -10 \
  --plan-only
```

---

## Command-Line Options

| Flag | Description |
|------|--------------|
| `--mode {resize, remove}` | select operation mode |
| `--image PATH` / `--output PATH` | input and output image paths |
| `--mask PATH` | protective mask (regions to **avoid** carving) |
| `--rmask PATH` | removal mask (regions to **delete**, required in `remove` mode) |
| `--dy`, `--dx` | vertical and horizontal seam deltas (negative = remove, positive = insert) |
| `--energy {forward, backward}` | energy function to use (forward by default) |
| `--hremove` | force horizontal seams for removal (overrides auto-orientation) |
| `--no-downsize` | disable pre-processing downscaling for speed |
| `--plan-only` | print planned operations and exit |
| `--viz-gif PATH.gif` | output animated GIF of seams |
| `--viz-every N` | record every N-th seam (default 1) |
| `--viz-max-frames K` | limit GIF frames (0 = unlimited) |
| `--viz-fps N` | frames per second for GIF (default 12) |

---

## Algorithm Summary

Seam carving works by:
1. Computing an **energy map** of the image (using gradient magnitude).
2. Finding the **lowest cumulative-energy seam** using dynamic programming.
3. Removing or inserting that seam.
4. Repeating for each requested seam count.

### Energy Functions
- **Backward energy** (simple gradient magnitude).  
- **Forward energy** (considers disruption to neighboring pixels — produces smoother results).  

### Object Removal
With a **removal mask**, seams are forced through marked pixels until the region disappears; then seams are **inserted back** to restore the original dimensions.

---

## Architecture (Refactored Design)

| File | Role |
|------|------|
| `cli.py` | parses CLI arguments; handles plan-only mode, auto-orientation, and visualization setup |
| `utils.py` | helper utilities: `Config` dataclass, resizing/rotation, mask preparation, I/O dtype management |
| `energy.py` | forward/backward energy functions |
| `seams.py` | seam detection, DP accumulation, and seam manipulation (remove/insert) |
| `ops.py` | high-level operations: accumulation & backtracking (Numba-accelerated), seam add/remove |
| `viz.py` | Optional GIF writer; draws seams, normalizes frame size with padding |
| `seam_carving.py` | lightweight entrypoint that delegates to `cli.main()` |

---

## Comparison to Original Implementation

| Area | Original | Refactored |
|------|----------|------------|
| **global state** | several module-level globals (`USE_FORWARD_ENERGY`, `SHOULD_DOWNSIZE`, etc.) | replaced by `Config` dataclass threaded through functions |
| **energy modes** | **forward energy by default** with `-backward_energy` toggle present | same modes, but configuration via `--energy` and safer JIT placement (see below) |
| **Numba usage** | `@jit` applied broadly (incl. functions that call OpenCV/SciPy) → falls back to object mode | `@njit(cache=True)` used **only** on pure DP loops (no cv2/scipy inside JIT) → reliable speedups |
| **visualization** | `-vis` flag opens an OpenCV window; shows seams in-place using `imshow`/`waitKey` | optional **GIF** export (`--viz-gif`) with fixed frame size via padding; non-blocking & scriptable |
| **CLI** | flags like `-resize`, `-remove`, `-backward_energy`, `-dy`, `-dx` | modern CLI (`--mode`, `--energy`, `--plan-only`, etc.) with **backward-compat aliases** preserved |
| **downsizing** | present: `SHOULD_DOWNSIZE=True`, `DOWNSIZE_WIDTH=500` (globals) | preserved, moved to `Config` + `--no-downsize`; masks are resized consistently |
| **mask handling** | thresholding inline inside seam finder; shape mismatches not validated; uses deprecated `np.bool`, `np.int` | centralized `prepare_mask()`: shape-checked, binarized once to `bool`; dtype modernized |
| **dry-run / planning** | not available | `--plan-only` prints mode, energy, sizes, downsizing decision, mask presence, orientation |
| **auto-orientation (remove)** | manual choice via `-hremove` | auto picks horizontal vs. vertical based on removal mask aspect ratio (override with `--hremove`) |
| **I/O & dtype** | mixed conversions; grayscale in forward energy handled ad hoc | contract: `uint8` at I/O, `float64` for compute; cast once at entry points |

---

## Attribution

The original implementation is at **[andrewdcampbell/seam-carving](https://github.com/andrewdcampbell/seam-carving)**, licensed under GNU General Public License v3.0.

This repository refactors and extends that implementation for clarity, modularity, and performance while maintaining educational and research compatibility.

---

## License

This refactored code is released under **GNU General Public License v3.0** (same as the original project).

---

## Author (Refactor & Enhancements)

**Pyait Myat**

---

### Key Additions in This Refactor
- `Config` dataclass replacing global state
- modular architecture (`cli.py`, `ops.py`, `energy.py`, `seams.py`, `viz.py`)
- safe, effective Numba acceleration on DP loops only
- auto-orientation for object removal
- `--plan-only` dry-run
- centralized, validated mask handling
- consistent dtype pipeline (`uint8` I/O, `float64` compute)
- GIF visualization with padding to keep frame sizes consistent
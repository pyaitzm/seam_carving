from __future__ import annotations

import argparse
import os
import sys
import cv2
import numpy as np

from utils import Config, resize, save_uint8, prepare_mask
from ops import seam_carve, object_removal
from viz import VizGifRecorder

AUTO_ORIENT_EPS = 1.2  # safety margin: treat mask as "wide" if w >= h*1.2


def parse_args() -> dict:
    ap = argparse.ArgumentParser(description="Content-aware seam carving / object removal")

    ap.add_argument(
        "--mode",
        choices=("resize", "remove"),
        required=False,  # keep False so legacy -resize/-remove still work
        help="Operation mode: resize (add/remove seams) or remove (object removal with rmask).",
    )

    ap.add_argument("-im", "--image", help="Path to input image", required=True)
    ap.add_argument("-out", "--output", help="Output image path", required=True)

    # Masks (add legacy aliases for back-compat)
    ap.add_argument("--mask", "-mask", help="Path to protective mask (areas to avoid carving)")
    ap.add_argument("--rmask", "-rmask", help="Path to removal mask (areas to delete) [required in remove mode]")

    # Seam counts (add legacy aliases)
    ap.add_argument("--dy", "-dy", type=int, default=0,
                    help="Vertical seams to add (+) / remove (-)")
    ap.add_argument("--dx", "-dx", type=int, default=0,
                    help="Horizontal seams to add (+) / remove (-)")

    # Object removal orientation
    ap.add_argument("--hremove", action="store_true",
                    help="Use horizontal seam orientation for object removal")

    # Energy choice
    ap.add_argument(
        "--energy",
        choices=("forward", "backward"),
        default="forward",
        help="Energy function to use (default: forward).",
    )

    # Downsizing toggle
    ap.add_argument(
        "--no-downsize",
        action="store_true",
        help="Disable default downsizing (processing is slower on large images).",
    )

    # Plan-only (dry run)
    ap.add_argument(
        "--plan-only",
        action="store_true",
        help="Print what would happen (mode, energy, dims, downsizing decision, masks) and exit without processing.",
    )

    # Visualization (GIF)
    ap.add_argument("--viz-gif", help="Path to an output GIF that visualizes carved seams over time.")
    ap.add_argument("--viz-every", type=int, default=1, help="Record every N-th seam (default: 1 = every seam).")
    ap.add_argument("--viz-max-frames", type=int, default=0, help="Optional cap on recorded frames (0 = unlimited).")
    ap.add_argument("--viz-fps", type=int, default=12, help="GIF frames per second (default: 12).")

    # Backward-compat shim (silent)
    ap.add_argument("-resize", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("-remove", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("-backward_energy", action="store_true", help=argparse.SUPPRESS)

    return vars(ap.parse_args())


def validate_and_normalize_args(a: dict) -> dict:
    # Back-compat mapping
    if a.get("mode") is None:
        if a.get("resize"):
            a["mode"] = "resize"
        elif a.get("remove"):
            a["mode"] = "remove"

    if a["energy"] == "forward" and a.get("backward_energy"):
        a["energy"] = "backward"

    # Strong validations
    if a.get("mode") not in ("resize", "remove"):
        sys.exit("Error: You must specify --mode {resize,remove} (or use legacy -resize / -remove).")

    if a["mode"] == "remove" and not a.get("rmask"):
        sys.exit("Error: --mode remove requires --rmask <path-to-removal-mask>.")
    if a["mode"] == "resize" and a.get("rmask"):
        sys.exit("Error: --mode resize forbids --rmask (did you mean --mask?).")

    # Basic file checks
    if not os.path.exists(a["image"]):
        sys.exit(f"Error: Input image not found: {a['image']}")
    if a.get("mask") and not os.path.exists(a["mask"]):
        sys.exit(f"Error: Protective mask not found: {a['mask']}")
    if a.get("rmask") and not os.path.exists(a["rmask"]):
        sys.exit(f"Error: Removal mask not found: {a['rmask']}")

    # Ensure output directory exists (quality-of-life)
    out_dir = os.path.dirname(os.path.abspath(a["output"]))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Visualization: ensure directory as well
    if a.get("viz_gif"):
        vg_dir = os.path.dirname(os.path.abspath(a["viz_gif"]))
        if vg_dir and not os.path.exists(vg_dir):
            os.makedirs(vg_dir, exist_ok=True)

    return a


def auto_orientation_from_mask_gray(rmask_gray: np.ndarray, threshold: int) -> tuple[str, bool]:
    """
    Decide orientation from the grayscale removal mask:
      - If mask is 'wide' (bbox width >= bbox height * AUTO_ORIENT_EPS) -> horizontal seams
      - Else -> vertical seams
    Returns: (orientation_str, auto_used_bool)
    """
    if rmask_gray is None:
        return "vertical", False  # default

    # Binarize quickly (matching prepare_mask threshold semantics)
    if rmask_gray.ndim == 3:
        rmask_gray = cv2.cvtColor(rmask_gray, cv2.COLOR_BGR2GRAY)
    if rmask_gray.dtype != np.uint8:
        rmask_gray = rmask_gray.astype(np.int32)

    rmask_bin = rmask_gray > int(threshold)

    if not np.any(rmask_bin):
        return "vertical", True

    ys, xs = np.where(rmask_bin)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    h = (y1 - y0 + 1)
    w = (x1 - x0 + 1)

    if w >= h * AUTO_ORIENT_EPS:
        return "horizontal", True
    else:
        return "vertical", True


def print_plan(args: dict, cfg: Config, im_work, mask_work, rmask_work, H: int, W: int, chosen_orientation: str, auto_used: bool) -> None:
    """Emit a deterministic, human-friendly plan and exit."""
    downsizing_decision = (cfg.should_downsize and im_work.shape[1] == cfg.downsize_width)
    working_w = im_work.shape[1]
    working_h = im_work.shape[0]

    print("=== Seam Carving Plan ===")
    print(f"Mode:            {args['mode']}")
    print(f"Energy:          {args['energy']}")
    print(f"Downsize:        {'ON' if cfg.should_downsize else 'OFF'} "
          f"({'will downsize' if downsizing_decision else 'no downsizing needed'})")
    print(f"Working size:    {working_w}x{working_h} (WxH)")

    if args["mode"] == "resize":
        final_w = W + args["dx"]
        final_h = H + args["dy"]
        print(f"Resize intents:  dy={args['dy']} (height), dx={args['dx']} (width)")
        print(f"Target size:     {final_w}x{final_h} (WxH)")
        print(f"Protect mask:    {'yes' if mask_work is not None else 'no'}")
    else:
        label = f"{chosen_orientation} (auto)" if auto_used and not args["hremove"] else chosen_orientation
        print(f"Removal mask:    {'yes' if rmask_work is not None else 'no'} (required)")
        print(f"Protect mask:    {'yes' if mask_work is not None else 'no'}")
        print(f"Orientation:     {label} seams for removal")
        print("Restoration:     Will add seams after removal to restore original dimensions.")

    # Visualization summary
    if args.get("viz_gif"):
        cap = args["viz_max_frames"] if args["viz_max_frames"] > 0 else "unlimited"
        print(f"Visualization:   GIF -> {args['viz_gif']}  (every={args['viz_every']}, max_frames={cap}, fps={args['viz_fps']})")
    else:
        print("Visualization:   (disabled)")

    print("Plan-only:       No processing will be performed.")
    sys.exit(0)


def main():
    args = parse_args()
    args = validate_and_normalize_args(args)

    # Read inputs
    im_u8 = cv2.imread(args["image"])
    assert im_u8 is not None, f"Could not read image at {args['image']}"
    mask_raw = cv2.imread(args["mask"], 0) if args.get("mask") else None
    rmask_raw = cv2.imread(args["rmask"], 0) if args.get("rmask") else None

    # Build Config
    cfg = Config(
        use_forward_energy=(args["energy"] == "forward"),
        should_downsize=(not args["no_downsize"]),
        downsize_width=500,
        energy_mask_const=100000.0,
        mask_threshold=10,
    )

    # Optional downsizing (prepare working arrays)
    h0, w0 = im_u8.shape[:2]
    if cfg.should_downsize and w0 > cfg.downsize_width:
        im_work = resize(im_u8, width=cfg.downsize_width)
        mask_work = resize(mask_raw, width=cfg.downsize_width) if mask_raw is not None else None
        rmask_work = resize(rmask_raw, width=cfg.downsize_width) if rmask_raw is not None else None
    else:
        im_work, mask_work, rmask_work = im_u8, mask_raw, rmask_raw

    H, W = im_work.shape[:2]

    # Decide orientation for remove mode (before binarization; used for plan and run)
    if args["mode"] == "remove":
        if args["hremove"]:
            chosen_orientation, auto_used = "horizontal", False
        else:
            chosen_orientation, auto_used = auto_orientation_from_mask_gray(rmask_work, cfg.mask_threshold)

    # Plan-only?
    if args["plan_only"]:
        if args["mode"] == "resize":
            print_plan(args, cfg, im_work, mask_work, rmask_work, H, W, chosen_orientation="vertical", auto_used=False)
        else:
            print_plan(args, cfg, im_work, mask_work, rmask_work, H, W, chosen_orientation=chosen_orientation, auto_used=auto_used)

    # Prepare boolean masks
    mask_bool = prepare_mask(mask_work, (H, W), cfg.mask_threshold, name="mask", allow_none=True)
    rmask_bool = None
    if args["mode"] == "remove":
        rmask_bool = prepare_mask(rmask_work, (H, W), cfg.mask_threshold, name="removal mask", allow_none=False)

    # Optional GIF recorder
    recorder = None
    if args.get("viz_gif"):
        max_frames = args["viz_max_frames"] if args["viz_max_frames"] > 0 else None
        recorder = VizGifRecorder(
            gif_path=args["viz_gif"],
            every=max(1, int(args["viz_every"])),
            max_frames=max_frames,
            fps=max(1, int(args["viz_fps"])),
        )

    # Execute
    if args["mode"] == "resize":
        output = seam_carve(im_work, args["dy"], args["dx"], cfg, mask_bool,
                            on_seam=(recorder.on_seam if recorder else None))
    else:  # remove
        use_horizontal = args["hremove"] or (chosen_orientation == "horizontal")
        output = object_removal(im_work, rmask_bool, cfg, mask_bool,
                                horizontal_removal=use_horizontal,
                                on_seam=(recorder.on_seam if recorder else None))

    # Save outputs
    save_uint8(args["output"], output)

    # Write GIF (if any)
    if recorder is not None:
        recorder.close()


if __name__ == "__main__":
    main()

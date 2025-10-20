#!/usr/bin/env python3
"""
Interactive SAM (clean display):
- p: switch to positive mode (green points)
- n: switch to negative mode (red points)
- Left-click: add a point in current mode
- ESC: save prompts JSON + binary mask .npy and exit

All instructions and status are printed to the terminal.
No text is drawn on the image window.
"""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch

try:
    from segment_anything import sam_model_registry, SamPredictor
except Exception as e:
    raise SystemExit(
        "Could not import 'segment_anything'.\n"
        "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git"
    ) from e


def load_rgb_bgr(image_path: str):
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, bgr


def overlay_mask_on_bgr(bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5):
    """
    Overlay a binary mask on a BGR image with simple alpha blending (green).
    mask: boolean or 0/1 array in image resolution (H, W).
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    overlay = bgr.copy()
    color = np.array([0, 255, 0], dtype=np.uint8)  # green in BGR
    overlay[mask] = (overlay[mask] * (1 - alpha) + color * alpha).astype(np.uint8)
    return overlay


def draw_points_on_bgr(img: np.ndarray, points, labels, scale: float):
    """
    Draw points as small circles on the display image (scaled).
    Positive (1): green; Negative (0): red.
    """
    disp = img.copy()
    for (x, y), lab in zip(points, labels):
        px = int(round(x * scale))
        py = int(round(y * scale))
        color = (0, 255, 0) if lab == 1 else (0, 0, 255)  # BGR
        cv2.circle(disp, (px, py), 5, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(disp, (px, py), 8, color, thickness=1, lineType=cv2.LINE_AA)
    return disp


def make_parser():
    p = argparse.ArgumentParser("Interactive SAM (clean): p/n to choose label, click to add, ESC to save.")
    p.add_argument("--image", type=str, required=True, help="Path to input image (jpg/png/...)")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to SAM checkpoint .pth")
    p.add_argument("--model-type", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b"],
                   help="SAM model type matching the checkpoint.")
    p.add_argument("--out-dir", type=str, default="./outputs", help="Directory to save outputs")
    p.add_argument("--max-window", type=int, default=1280, help="Max display dimension (preserves aspect).")
    p.add_argument("--multimask", action="store_true",
                   help="If set, let SAM produce 3 masks and pick the best by score.")
    return p


def main():
    args = make_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load SAM
    print(f"[INFO] Loading SAM '{args.model_type}' from: {args.checkpoint}")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Load image
    img_rgb, img_bgr = load_rgb_bgr(args.image)
    H, W = img_bgr.shape[:2]

    # Fit into window while keeping aspect ratio
    scale = min(args.max_window / max(H, W), 1.0)
    disp_size = (int(round(W * scale)), int(round(H * scale)))

    # Prepare predictor
    print(f"[INFO] Encoding image (longest side -> 1024, padded to square for encoder).")
    predictor.set_image(img_rgb)

    # State
    points = []         # [(x, y), ...] in ORIGINAL coords
    labels = []         # [1 or 0, ...]
    t_counter = 0
    mode_label = 1      # 1 = positive (default), 0 = negative
    current_mask = np.zeros((H, W), dtype=bool)
    window_name = "SAM Interactive (clean)"

    # Mouse callback context
    ctx = {
        "points": points,
        "labels": labels,
        "t_counter": t_counter,
        "mode_label": mode_label,
        "mask": current_mask,
        "dirty": True,  # forces initial draw
    }

    def recompute_mask():
        if len(ctx["points"]) == 0:
            ctx["mask"] = np.zeros((H, W), dtype=bool)
            return
        pc = np.array(ctx["points"], dtype=np.float32)
        pl = np.array(ctx["labels"], dtype=np.int32)
        with torch.inference_mode():
            masks, scores, _ = predictor.predict(
                point_coords=pc,
                point_labels=pl,
                multimask_output=args.multimask
            )
        if args.multimask:
            best_idx = int(np.argmax(scores))
            ctx["mask"] = masks[best_idx]
        else:
            ctx["mask"] = masks[0]

    def print_mode(m):
        print(f"[MODE] {'POSITIVE (p)'.ljust(14) if m == 1 else 'NEGATIVE (n)'.ljust(14)} | "
              f"t={ctx['t_counter']}  pos={sum(1 for l in ctx['labels'] if l==1)}  "
              f"neg={sum(1 for l in ctx['labels'] if l==0)}")

    def on_mouse(event, x, y, flags, param):
        # Left click adds a point in current mode_label (scaled -> original coords)
        if event == cv2.EVENT_LBUTTONDOWN:
            x0 = int(round(x / scale))
            y0 = int(round(y / scale))
            x0 = np.clip(x0, 0, W - 1)
            y0 = np.clip(y0, 0, H - 1)
            ctx["points"].append((x0, y0))
            ctx["labels"].append(ctx["mode_label"])
            ctx["t_counter"] += 1
            recompute_mask()
            pos_count = sum(1 for l in ctx["labels"] if l == 1)
            neg_count = sum(1 for l in ctx["labels"] if l == 0)
            print(f"[CLICK] t={ctx['t_counter']:>3}  "
                  f"({'POS' if ctx['mode_label']==1 else 'NEG'})  x={x0} y={y0}  "
                  f"| totals -> pos={pos_count}, neg={neg_count}")
            ctx["dirty"] = True

    print("[INFO] Controls: p=positive, n=negative, Left-click=add point, ESC=save & exit")
    print_mode(mode_label)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(window_name, disp_size[0], disp_size[1])
    cv2.setMouseCallback(window_name, on_mouse)

    # Main UI loop
    while True:
        if ctx["dirty"]:
            # Compose display (no HUD text)
            base = overlay_mask_on_bgr(img_bgr, ctx["mask"], alpha=0.5)
            base = cv2.resize(base, disp_size, interpolation=cv2.INTER_LINEAR)
            base = draw_points_on_bgr(base, ctx["points"], ctx["labels"], scale)
            cv2.imshow(window_name, base)
            ctx["dirty"] = False

        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('p'):
            ctx["mode_label"] = 1
            print_mode(ctx["mode_label"])
        elif key == ord('n'):
            ctx["mode_label"] = 0
            print_mode(ctx["mode_label"])
        elif key == ord('r'):
            # Optional reset
            ctx["points"].clear()
            ctx["labels"].clear()
            ctx["t_counter"] = 0
            recompute_mask()
            print("[RESET] Cleared all prompts.")
            print_mode(ctx["mode_label"])
            ctx["dirty"] = True

    cv2.destroyAllWindows()

    # Save outputs
    stem = Path(args.image).stem
    prompts_path = out_dir / f"{stem}_prompts.json"
    mask_path = out_dir / f"{stem}_mask.npy"

    prompts_out = {
        "prompts": [
            {"t": i + 1, "x": int(x), "y": int(y), "label": int(lab)}
            for i, ((x, y), lab) in enumerate(zip(ctx["points"], ctx["labels"]))
        ]
    }
    with open(prompts_path, "w") as f:
        json.dump(prompts_out, f, indent=2)

    # Save binary mask (0/1)
    bin_mask = (ctx["mask"].astype(np.uint8) > 0).astype(np.uint8)
    np.save(mask_path, bin_mask)

    print(f"[SAVED] prompts -> {prompts_path}")
    print(f"[SAVED] binary mask (0/1) -> {mask_path}")
    print("[DONE] Goodbye.")


if __name__ == "__main__":
    main()

"""
Simple depth visualization for ARTi4D depth maps.

Usage:
    python visualize_depth.py --depth_dir  /path/to/depth/
    python visualize_depth.py --depth_file /path/to/depth_image_123.png
    python visualize_depth.py --depth_dir  /path/to/depth/ --colormap jet --max_depth 5.0
"""

import argparse
import cv2
import numpy as np
from pathlib import Path


COLORMAPS = {
    "turbo":   cv2.COLORMAP_TURBO,
    "jet":     cv2.COLORMAP_JET,
    "plasma":  cv2.COLORMAP_PLASMA,
    "inferno": cv2.COLORMAP_INFERNO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "magma":   cv2.COLORMAP_MAGMA,
}


def depth_to_color(depth_mm: np.ndarray, max_depth_m: float, colormap: int) -> np.ndarray:
    """uint16 mm depth  →  colourised uint8 BGR, black = invalid."""
    depth_m = depth_mm.astype(np.float32) / 1000.0
    valid   = (depth_m > 0) & (depth_m < max_depth_m)

    norm         = np.zeros_like(depth_m)
    norm[valid]  = depth_m[valid] / max_depth_m     # 0–1
    norm         = (norm * 255).astype(np.uint8)

    colored         = cv2.applyColorMap(norm, colormap)
    colored[~valid] = 0                             # black = missing
    return colored


def overlay_stats(img: np.ndarray, depth_mm: np.ndarray, label: str) -> np.ndarray:
    depth_m = depth_mm.astype(np.float32) / 1000.0
    valid   = depth_m[depth_m > 0]
    total   = depth_mm.size

    lines = [
        label,
        f"valid : {len(valid):,} / {total:,}  ({100*len(valid)/total:.1f}%)",
        f"min   : {valid.min():.3f} m"  if len(valid) else "min  : -",
        f"max   : {valid.max():.3f} m"  if len(valid) else "max  : -",
        f"mean  : {valid.mean():.3f} m" if len(valid) else "mean : -",
    ]
    out = img.copy()
    for i, line in enumerate(lines):
        y = 22 + i * 22
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0),       1, cv2.LINE_AA)
    return out


def show_single(path: Path, max_depth_m: float, cmap_key: str):
    depth_mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        print(f"[ERROR] Could not read {path}")
        return
    colored = depth_to_color(depth_mm, max_depth_m, COLORMAPS[cmap_key])
    colored = overlay_stats(colored, depth_mm, path.name)
    cv2.imshow("Depth viewer  [q] quit", colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_directory(depth_dir: Path, max_depth_m: float, cmap_key: str):
    files = sorted(depth_dir.glob("depth_image_*.png"))
    if not files:
        files = sorted(depth_dir.glob("*.png"))
    if not files:
        print(f"[ERROR] No PNG files found in {depth_dir}")
        return

    print(f"Found {len(files)} depth images.")
    print("  [n] / →   next          [p] / ←   prev")
    print("  [s]       save frame    [q] / ESC  quit")

    idx = 0
    win = "Depth viewer  [n/p] navigate  [s] save  [q] quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        path     = files[idx]
        depth_mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        colored  = depth_to_color(depth_mm, max_depth_m, COLORMAPS[cmap_key])
        colored  = overlay_stats(colored, depth_mm, f"[{idx+1}/{len(files)}]  {path.name}")

        cv2.imshow(win, colored)
        key = cv2.waitKey(0) & 0xFF

        if   key in (ord("q"), 27):        # quit
            break
        elif key in (ord("n"), 83, 100):   # next  ( n / → / d )
            idx = min(idx + 1, len(files) - 1)
        elif key in (ord("p"), 81, 97):    # prev  ( p / ← / a )
            idx = max(idx - 1, 0)
        elif key == ord("s"):
            out_path = path.with_name(path.stem + "_vis.png")
            cv2.imwrite(str(out_path), colored)
            print(f"  Saved → {out_path}")

    cv2.destroyAllWindows()


def main():
    ap  = argparse.ArgumentParser(description="Visualize ARTi4D depth maps")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--depth_dir",  type=Path, help="Directory of depth PNGs")
    grp.add_argument("--depth_file", type=Path, help="Single depth PNG")
    ap.add_argument("--max_depth",   type=float, default=3.0,
                    help="Colour scale ceiling in metres (default: 3.0)")
    ap.add_argument("--colormap",    default="turbo", choices=COLORMAPS.keys(),
                    help="Colour map (default: turbo)")
    args = ap.parse_args()

    if args.depth_file:
        show_single(args.depth_file, args.max_depth, args.colormap)
    else:
        show_directory(args.depth_dir, args.max_depth, args.colormap)


if __name__ == "__main__":
    main()
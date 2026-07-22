"""Export demo video: camera + floor grid side-by-side or stacked.

Usage:
  python export_demo_video.py
  python export_demo_video.py --layout horizontal --height 540
  python export_demo_video.py --layout vertical --width 800

README 建議用 WebP（比 GIF 小、畫質好）；可用 ffmpeg 從 mp4 轉：
  ffmpeg -i test/demo_detect_grid.mp4 -vf "fps=12,scale=720:-1" -loop 0 test/demo_detect_grid.webp
  ffmpeg -i test/demo_detect_grid.mp4 -vf "fps=10,scale=640:-1" test/demo_detect_grid.gif
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from detect_grid import (
    CellStabilizer,
    detect_and_locate,
    load_homography,
    render_detection_view,
    resize_for_preview,
)

DEFAULT_SOURCE = Path(__file__).resolve().parent / "test" / "test.mp4"
DEFAULT_CALIB = Path(__file__).resolve().parent / "calibration" / "homography.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "test" / "demo_detect_grid.mp4"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export cam+grid demo video")
    p.add_argument("--source", default=str(DEFAULT_SOURCE))
    p.add_argument("--calib", default=str(DEFAULT_CALIB))
    p.add_argument("--model", default="yolo26s.pt")
    p.add_argument("--conf", type=float, default=0.45)
    p.add_argument("--ref", choices=["auto", "foot", "head_drop"], default="auto")
    p.add_argument("--aspect", type=float, default=3.0)
    p.add_argument("--truncate-ratio", type=float, default=1.6)
    p.add_argument("--min-h-ratio", type=float, default=0.12)
    p.add_argument("--min-aspect", type=float, default=1.15)
    p.add_argument("--min-bottom-ratio", type=float, default=0.28)
    p.add_argument("--valid-xmin", type=float, default=170.0)
    p.add_argument(
        "--layout",
        choices=["horizontal", "vertical"],
        default="horizontal",
        help="horizontal: cam left + grid right; vertical: cam top + grid bottom",
    )
    p.add_argument("--width", type=int, default=800, help="output width for vertical layout (px)")
    p.add_argument("--height", type=int, default=720, help="panel height for horizontal layout (px)")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--no-timing", action="store_true")
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="run YOLO every N frames; skipped frames reuse last detections",
    )
    p.add_argument(
        "--cell-hold",
        type=int,
        default=2,
        help="a cell only lights/clears after N consecutive DETECTION RUNS agree "
        "(counted in stride units, not raw frames); 1 disables debounce",
    )
    return p.parse_args()


def fit_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == width:
        return img
    scale = width / float(w)
    return cv2.resize(img, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def fit_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == height:
        return img
    scale = height / float(h)
    return cv2.resize(img, (max(1, int(round(w * scale))), height), interpolation=cv2.INTER_AREA)


def stack_vertical(cam: np.ndarray, grid: np.ndarray, width: int) -> np.ndarray:
    top = fit_width(cam, width)
    bottom = fit_width(grid, width)
    sep = np.full((2, width, 3), (40, 40, 40), dtype=np.uint8)
    return np.vstack([top, sep, bottom])


def stack_horizontal(cam: np.ndarray, grid: np.ndarray, height: int) -> np.ndarray:
    left = fit_height(cam, height)
    right = fit_height(grid, height)
    sep = np.full((height, 2, 3), (40, 40, 40), dtype=np.uint8)
    return np.hstack([left, sep, right])


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"找不到來源：{source}")
    calib_path = Path(args.calib)
    if not calib_path.exists():
        raise SystemExit(f"找不到校正檔：{calib_path}")

    h_mat = load_homography(calib_path)
    model = YOLO(args.model)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"無法開啟影片：{source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.stride < 1:
        raise SystemExit("--stride 必須 >= 1")

    det_kw = dict(
        conf=args.conf,
        ref=args.ref,
        aspect=args.aspect,
        truncate_ratio=args.truncate_ratio,
        min_h_ratio=args.min_h_ratio,
        min_aspect=args.min_aspect,
        min_bottom_ratio=args.min_bottom_ratio,
    )

    stabilizer = CellStabilizer(args.cell_hold)
    confirmed_cells: set[tuple[int, int]] = set()

    writer: cv2.VideoWriter | None = None
    n = 0
    frame_idx = 0
    last_dets: list[dict] = []
    last_timing: tuple[float, float] | None = None
    detect_runs = 0
    t0 = time.perf_counter()
    layout_desc = f"height={args.height}" if args.layout == "horizontal" else f"width={args.width}"
    stride_note = f"，stride={args.stride}" if args.stride > 1 else ""
    print(f"匯出中：{source.name} → {out_path}（約 {total} 幀，{args.layout}，{layout_desc}{stride_note}）")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_idx += 1
            run_detect = frame_idx == 1 or (frame_idx - 1) % args.stride == 0
            if run_detect:
                last_dets, detect_ms, locate_ms = detect_and_locate(
                    frame, model, h_mat, **det_kw
                )
                last_timing = None if args.no_timing else (detect_ms, locate_ms)
                detect_runs += 1
                raw_cells = {d["cell"] for d in last_dets if d.get("cell") is not None}
                confirmed_cells = stabilizer.update(raw_cells)

            timing = last_timing if not args.no_timing else None
            vis, grid, _logs = render_detection_view(
                frame,
                last_dets,
                h_mat,
                args.valid_xmin,
                timing=timing,
                cached=not run_detect,
                grid_cells=confirmed_cells,
            )
            # Horizontal: keep full-res cam so the README embed fills the column width
            if args.layout == "horizontal":
                cam = vis
            else:
                cam = resize_for_preview(vis, max(args.width, 960))

            if args.layout == "horizontal":
                stacked = stack_horizontal(cam, grid, args.height)
            else:
                stacked = stack_vertical(cam, grid, args.width)
            if writer is None:
                h, w = stacked.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
                if not writer.isOpened():
                    raise SystemExit(f"無法建立輸出影片：{out_path}")
                print(f"輸出解析度：{w}x{h} @ {fps:.2f} fps")
            writer.write(stacked)
            n += 1
            if n % 50 == 0 or (total and n == total):
                print(f"  {n}/{total or '?'} 幀")
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - t0
    size_mb = out_path.stat().st_size / (1024 * 1024) if out_path.exists() else 0
    print(
        f"完成：{out_path}（{n} 幀，YOLO {detect_runs} 次，"
        f"{size_mb:.1f} MB，耗時 {elapsed:.1f}s）"
    )


if __name__ == "__main__":
    main()

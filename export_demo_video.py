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
    annotate_and_cells,
    draw_multi_grid,
    extract_foot_detections,
    image_to_world,
    load_homography,
    resize_for_preview,
)
from grid_occupancy import world_to_cell

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

    writer: cv2.VideoWriter | None = None
    n = 0
    t0 = time.perf_counter()
    layout_desc = f"height={args.height}" if args.layout == "horizontal" else f"width={args.width}"
    print(f"匯出中：{source.name} → {out_path}（約 {total} 幀，{args.layout}，{layout_desc}）")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            td0 = time.perf_counter()
            results = model.predict(frame, conf=args.conf, classes=[0], verbose=False)
            detect_ms = (time.perf_counter() - td0) * 1000.0

            tl0 = time.perf_counter()
            dets = extract_foot_detections(
                results[0],
                args.conf,
                frame.shape[0],
                frame.shape[1],
                mode=args.ref,
                aspect=args.aspect,
                truncate_ratio=args.truncate_ratio,
                min_h_ratio=args.min_h_ratio,
                min_aspect=args.min_aspect,
                min_bottom_ratio=args.min_bottom_ratio,
            )
            for det in dets:
                fx, fy = det["foot"]
                wx, wy = image_to_world(h_mat, fx, fy)
                det["world"] = (wx, wy)
                det["cell"] = world_to_cell(wx, wy)
            locate_ms = (time.perf_counter() - tl0) * 1000.0

            vis, cells, _logs = annotate_and_cells(frame, dets, h_mat, args.valid_xmin)
            grid = draw_multi_grid(cells, args.valid_xmin)
            # Horizontal: keep full-res cam so the README embed fills the column width
            if args.layout == "horizontal":
                cam = vis
            else:
                cam = resize_for_preview(vis, max(args.width, 960))

            if not args.no_timing and dets:
                from detect_grid import put_label

                timing_txt = f"detect {detect_ms:.0f}ms  locate {locate_ms:.2f}ms"
                (tw, _th), _ = cv2.getTextSize(
                    timing_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2
                )
                tx = max(10, grid.shape[1] - tw - 20)
                put_label(
                    grid,
                    timing_txt,
                    (tx, 30),
                    fg=(0, 255, 255),
                    bg=(0, 0, 0),
                    scale=0.85,
                    thickness=2,
                )

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
    print(f"完成：{out_path}（{n} 幀，{size_mb:.1f} MB，耗時 {elapsed:.1f}s）")


if __name__ == "__main__":
    main()

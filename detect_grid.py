"""Detect person with YOLO, estimate ground contact, light grid cell.

Ground reference:
  - foot: bbox bottom-center (when ankles visible)
  - head_drop: from head (bbox top) drop down by estimated body height in image,
    then map that pixel with floor Homography
  - auto (default): foot when bbox looks full; head_drop when likely truncated

Usage:
  python detect_grid.py --source test/test.mp4
  python detect_grid.py --source test/test.mp4 --ref auto
  python detect_grid.py --source test/test.mp4 --ref head_drop
  python detect_grid.py --source test/test.mp4 --ref foot
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from grid_occupancy import (
    X_EDGES,
    cell_label,
    draw_grid,
    imread_unicode,
    imwrite_unicode,
    world_to_cell,
)

DEFAULT_IMAGE = Path(__file__).resolve().parent / "test" / "static_frame.jpg"
DEFAULT_CALIB = Path(__file__).resolve().parent / "calibration" / "homography.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "test" / "detect_grid_preview.jpg"


def resize_for_preview(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if max_width <= 0 or w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def load_homography(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return np.array(payload["homography"], dtype=np.float64)


def image_to_world(h_mat: np.ndarray, x: float, y: float) -> tuple[float, float]:
    pts = np.array([[[x, y]]], dtype=np.float64)
    world = cv2.perspectiveTransform(pts, h_mat)[0, 0]
    return float(world[0]), float(world[1])


def open_capture(source: str) -> cv2.VideoCapture | None:
    if source.lower().startswith("rtsp://"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap if cap.isOpened() else None
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        return cap if cap.isOpened() else None
    path = Path(source)
    if path.exists() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        cap = cv2.VideoCapture(str(path))
        return cap if cap.isOpened() else None
    return None


def estimate_ref_point(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_h: int,
    mode: str,
    aspect: float,
    truncate_ratio: float,
) -> tuple[float, float, str]:
    """Return (ref_x, ref_y, used_mode).

    - foot: bbox bottom-center (true when ankles visible)
    - head_drop: from head (bbox top-center) drop down by estimated full-body height
      in image pixels, then apply floor Homography on that estimated ground pixel
    - auto: use head_drop when bbox looks truncated (short height vs width)
    """
    cx = 0.5 * (x1 + x2)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    head_x, head_y = cx, float(y1)
    foot_x, foot_y = cx, float(y2)

    looks_truncated = (bh / bw) < truncate_ratio

    if mode == "foot":
        return foot_x, min(foot_y, float(frame_h - 1)), "foot"
    if mode == "head_drop":
        est_h = bw * aspect
        est_foot_y = head_y + est_h
        return head_x, min(est_foot_y, float(frame_h - 1)), "head_drop"

    # auto
    if looks_truncated:
        est_h = bw * aspect
        est_foot_y = max(foot_y, head_y + est_h)
        return head_x, min(est_foot_y, float(frame_h - 1)), "head_drop"
    return foot_x, min(foot_y, float(frame_h - 1)), "foot"


def extract_foot_detections(
    result,
    conf_thres: float,
    frame_h: int,
    mode: str = "auto",
    aspect: float = 3.0,
    truncate_ratio: float = 1.6,
) -> list[dict]:
    """Return person ground-ref points from YOLO (COCO class 0)."""
    out: list[dict] = []
    if result.boxes is None or len(result.boxes) == 0:
        return out
    boxes = result.boxes
    for i in range(len(boxes)):
        if int(boxes.cls[i].item()) != 0:
            continue
        conf = float(boxes.conf[i].item())
        if conf < conf_thres:
            continue
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        ref_x, ref_y, used = estimate_ref_point(
            x1, y1, x2, y2, frame_h, mode, aspect, truncate_ratio
        )
        out.append(
            {
                "xyxy": (int(x1), int(y1), int(x2), int(y2)),
                "foot": (ref_x, ref_y),
                "head": (0.5 * (x1 + x2), float(y1)),
                "mode": used,
                "conf": conf,
            }
        )
    return out


def annotate_and_cells(
    frame: np.ndarray,
    detections: list[dict],
    h_mat: np.ndarray,
    valid_xmin: float,
) -> tuple[np.ndarray, set[tuple[int, int]], list[str]]:
    vis = frame.copy()
    cells: set[tuple[int, int]] = set()
    logs: list[str] = []

    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        fx, fy = det["foot"]
        hx, hy = det.get("head", (fx, y1))
        used = det.get("mode", "foot")
        wx, wy = image_to_world(h_mat, fx, fy)
        cell = world_to_cell(wx, wy)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # head marker
        cv2.circle(vis, (int(hx), int(hy)), 5, (255, 128, 0), -1)
        # estimated / measured ground ref
        color = (0, 0, 255) if used == "foot" else (255, 0, 255)
        cv2.circle(vis, (int(fx), int(fy)), 7, color, -1)
        cv2.line(vis, (int(hx), int(hy)), (int(fx), int(fy)), color, 1, cv2.LINE_AA)
        cv2.putText(
            vis,
            used,
            (int(fx) + 8, int(fy) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

        label = f"{det['conf']:.2f} ({wx:.0f},{wy:.0f})cm"
        low_conf = used == "head_drop"
        if cell is None:
            label += " OUT"
        else:
            cells.add(cell)
            label += f" c{cell[0]},r{cell[1]}"
            if X_EDGES[cell[0] + 1] <= valid_xmin:
                low_conf = True
                label += " LOW"
            logs.append(
                f"{used}=({fx:.1f},{fy:.1f}) world=({wx:.1f},{wy:.1f}) {cell_label(*cell)}"
                + (" [low]" if low_conf else "")
            )

        cv2.putText(
            vis,
            label,
            (x1, max(28, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return vis, cells, logs


def draw_multi_grid(cells: set[tuple[int, int]], valid_xmin: float) -> np.ndarray:
    """Light all occupied cells (draw base then overlay each)."""
    if not cells:
        return draw_grid(None, valid_x_min=valid_xmin)
    # draw by temporarily activating one-by-one on copies then merge max brightness
    base = draw_grid(None, valid_x_min=valid_xmin)
    for cell in cells:
        lit = draw_grid(cell, valid_x_min=valid_xmin)
        # where lit cell is yellow-ish, keep it
        mask = np.any(lit != base, axis=2)
        base[mask] = lit[mask]
    return base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO ground-ref point -> light floor grid")
    p.add_argument("--source", default=str(DEFAULT_IMAGE))
    p.add_argument("--calib", default=str(DEFAULT_CALIB))
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument(
        "--ref",
        choices=["auto", "foot", "head_drop"],
        default="auto",
        help="auto: foot when bbox looks full; head_drop when likely truncated",
    )
    p.add_argument(
        "--aspect",
        type=float,
        default=3.0,
        help="for head_drop: estimated full-body height ≈ bbox_width * aspect",
    )
    p.add_argument(
        "--truncate-ratio",
        type=float,
        default=1.6,
        help="auto switches to head_drop when bbox_h/bbox_w < this",
    )
    p.add_argument("--valid-xmin", type=float, default=170.0)
    p.add_argument("--max-width", type=int, default=1280)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    calib_path = Path(args.calib)
    if not calib_path.exists():
        raise SystemExit(f"找不到校正檔：{calib_path}")
    h_mat = load_homography(calib_path)
    model = YOLO(args.model)

    source = args.source
    is_image = Path(source).exists() and Path(source).suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    cam_win = "Detect + Ground Ref"
    grid_win = "Grid"
    cv2.namedWindow(cam_win, cv2.WINDOW_NORMAL)
    cv2.namedWindow(grid_win, cv2.WINDOW_NORMAL)
    print(f"參考點模式：{args.ref}（紅=foot，紫=head_drop）。按 q 結束，s 存圖。")

    def process_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        results = model.predict(frame, conf=args.conf, classes=[0], verbose=False)
        dets = extract_foot_detections(
            results[0],
            args.conf,
            frame.shape[0],
            mode=args.ref,
            aspect=args.aspect,
            truncate_ratio=args.truncate_ratio,
        )
        vis, cells, logs = annotate_and_cells(frame, dets, h_mat, args.valid_xmin)
        for line in logs:
            print(line)
        grid = draw_multi_grid(cells, args.valid_xmin)
        return vis, grid

    if is_image:
        frame = imread_unicode(Path(source))
        if frame is None:
            raise SystemExit(f"無法讀取影像：{source}")
        vis, grid = process_frame(frame)
        view = resize_for_preview(vis, args.max_width)
        while True:
            cv2.imshow(cam_win, view)
            cv2.imshow(grid_win, grid)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("s"):
                imwrite_unicode(Path(args.out), view)
                imwrite_unicode(Path(args.out).with_name("detect_grid_cells.jpg"), grid)
                print(f"已存：{args.out}")
            elif key in (ord("q"), 27):
                break
        cv2.destroyAllWindows()
        return

    cap = open_capture(source)
    if cap is None:
        raise SystemExit(f"無法開啟來源：{source}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("讀取結束或失敗。")
                break
            vis, grid = process_frame(frame)
            view = resize_for_preview(vis, args.max_width)
            cv2.imshow(cam_win, view)
            cv2.imshow(grid_win, grid)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                imwrite_unicode(Path(args.out), view)
                imwrite_unicode(Path(args.out).with_name("detect_grid_cells.jpg"), grid)
                print(f"已存：{args.out}")
            elif key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

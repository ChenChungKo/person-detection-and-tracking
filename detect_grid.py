"""Detect person (YOLO) + ground ref + light grid cell.

Ground reference:
  - foot / head_drop / auto (see --ref)

Usage:
  python detect_grid.py --source test/test.mp4 --ref auto
  python detect_grid.py --source test/test.mp4 --stride 3
  python detect_grid.py --source "rtsp://user:pass@ip:554/stream1" --ref auto
"""

from __future__ import annotations

import argparse
import json
import os
import time
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
    show_fixed_window,
    show_grid_window,
    world_to_cell,
)
from latest_frame import LatestFrameCapture

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


def is_plausible_person_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_h: int,
    frame_w: int,
    min_h_ratio: float = 0.12,
    min_aspect: float = 1.15,
    min_bottom_ratio: float = 0.28,
) -> bool:
    """Reject common desk/monitor false positives.

    Monitors often yield small, squarish boxes floating mid-frame.
    Real people (even seated) tend to be taller than wide and reach lower in the image.
    """
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    if bh < min_h_ratio * frame_h:
        return False
    if (bh / bw) < min_aspect:
        return False
    # box bottom should not sit high in the frame (typical monitor FP region)
    if y2 < min_bottom_ratio * frame_h:
        return False
    # discard tiny area relative to frame
    if (bw * bh) < 0.005 * frame_w * frame_h:
        return False
    return True


def extract_foot_detections(
    result,
    conf_thres: float,
    frame_h: int,
    frame_w: int,
    mode: str = "auto",
    aspect: float = 3.0,
    truncate_ratio: float = 1.6,
    min_h_ratio: float = 0.12,
    min_aspect: float = 1.15,
    min_bottom_ratio: float = 0.28,
) -> list[dict]:
    """Return person ground-ref points from YOLO detect boxes."""
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
        if not is_plausible_person_box(
            x1,
            y1,
            x2,
            y2,
            frame_h,
            frame_w,
            min_h_ratio=min_h_ratio,
            min_aspect=min_aspect,
            min_bottom_ratio=min_bottom_ratio,
        ):
            continue
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


def put_label(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    fg: tuple[int, int, int] = (0, 255, 255),
    bg: tuple[int, int, int] = (0, 0, 0),
    scale: float = 1.0,
    thickness: int = 2,
) -> None:
    """High-contrast label with filled background so text stays readable after preview resize."""
    x, y = org
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = 6
    x1 = max(0, x - pad)
    y1 = max(0, y - th - pad)
    x2 = min(img.shape[1] - 1, x + tw + pad)
    y2 = min(img.shape[0] - 1, y + baseline + pad)
    cv2.rectangle(img, (x1, y1), (x2, y2), bg, -1)
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        fg,
        thickness,
        cv2.LINE_AA,
    )


def annotate_and_cells(
    frame: np.ndarray,
    detections: list[dict],
    h_mat: np.ndarray,
    valid_xmin: float,
) -> tuple[np.ndarray, set[tuple[int, int]], list[str]]:
    vis = frame.copy()
    cells: set[tuple[int, int]] = set()
    logs: list[str] = []
    fs = max(1.0, frame.shape[1] / 1280.0)
    thick = max(2, int(round(2 * fs)))

    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        fx, fy = det["foot"]
        used = det.get("mode", "foot")
        box_color = (0, 255, 0)
        if "world" in det and "cell" in det:
            wx, wy = det["world"]
            cell = det["cell"]
        else:
            wx, wy = image_to_world(h_mat, fx, fy)
            cell = world_to_cell(wx, wy)

        cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, max(2, thick))
        ref_color = (0, 0, 255) if used == "foot" else (255, 0, 255)
        cv2.circle(vis, (int(fx), int(fy)), max(7, int(6 * fs)), ref_color, -1)

        if cell is None:
            low_conf = True
            label_y = y1 - 12
            if label_y < int(40 * fs):
                label_y = y1 + int(36 * fs)
            put_label(
                vis,
                "OUT",
                (x1, label_y),
                fg=(255, 255, 255),
                bg=(0, 0, 220),
                scale=0.9 * fs,
                thickness=thick,
            )
            logs.append(
                f"person {det['conf']:.2f} {used}=({fx:.1f},{fy:.1f}) "
                f"world=({wx:.1f},{wy:.1f}) OUT"
            )
        else:
            cells.add(cell)
            low_conf = used == "head_drop"
            if valid_xmin > 0 and X_EDGES[cell[0] + 1] <= valid_xmin:
                low_conf = True
            logs.append(
                f"person {det['conf']:.2f} {used}=({fx:.1f},{fy:.1f}) "
                f"world=({wx:.1f},{wy:.1f}) {cell_label(*cell)}"
                + (" [low]" if low_conf else "")
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


class CellStabilizer:
    """Debounce grid-cell occupancy against per-detection jitter.

    A cell only lights up after appearing in ``hold`` consecutive detection
    RUNS (not rendered/cached frames), and only turns off after being absent
    for ``hold`` consecutive runs. This keeps small bbox jitter (e.g. a
    slight body twist) from flickering between adjacent cells.
    """

    def __init__(self, hold: int = 2) -> None:
        self.hold = max(1, hold)
        self._on_streak: dict[tuple[int, int], int] = {}
        self._off_streak: dict[tuple[int, int], int] = {}
        self._confirmed: set[tuple[int, int]] = set()

    def update(self, raw_cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
        tracked = set(self._on_streak) | set(self._off_streak) | raw_cells | self._confirmed
        for cell in tracked:
            if cell in raw_cells:
                self._on_streak[cell] = self._on_streak.get(cell, 0) + 1
                self._off_streak[cell] = 0
                if self._on_streak[cell] >= self.hold:
                    self._confirmed.add(cell)
            else:
                self._off_streak[cell] = self._off_streak.get(cell, 0) + 1
                self._on_streak[cell] = 0
                if self._off_streak[cell] >= self.hold:
                    self._confirmed.discard(cell)
        # Drop fully-idle cells so the dicts do not grow without bound.
        for cell in list(self._on_streak):
            if self._on_streak[cell] == 0 and cell not in self._confirmed:
                self._on_streak.pop(cell, None)
                self._off_streak.pop(cell, None)
        return set(self._confirmed)


def detect_and_locate(
    frame: np.ndarray,
    model: YOLO,
    h_mat: np.ndarray,
    conf: float,
    ref: str,
    aspect: float,
    truncate_ratio: float,
    min_h_ratio: float,
    min_aspect: float,
    min_bottom_ratio: float,
) -> tuple[list[dict], float, float]:
    t0 = time.perf_counter()
    results = model.predict(frame, conf=conf, classes=[0], verbose=False)
    detect_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    dets = extract_foot_detections(
        results[0],
        conf,
        frame.shape[0],
        frame.shape[1],
        mode=ref,
        aspect=aspect,
        truncate_ratio=truncate_ratio,
        min_h_ratio=min_h_ratio,
        min_aspect=min_aspect,
        min_bottom_ratio=min_bottom_ratio,
    )
    for det in dets:
        fx, fy = det["foot"]
        wx, wy = image_to_world(h_mat, fx, fy)
        det["world"] = (wx, wy)
        det["cell"] = world_to_cell(wx, wy)
    locate_ms = (time.perf_counter() - t1) * 1000.0
    return dets, detect_ms, locate_ms


def render_detection_view(
    frame: np.ndarray,
    dets: list[dict],
    h_mat: np.ndarray,
    valid_xmin: float,
    timing: tuple[float, float] | None = None,
    cached: bool = False,
    grid_cells: set[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    vis, cells, logs = annotate_and_cells(frame, dets, h_mat, valid_xmin)
    display_cells = grid_cells if grid_cells is not None else cells
    grid = draw_multi_grid(display_cells, valid_xmin)

    if timing is not None:
        detect_ms, locate_ms = timing
        suffix = "  cached" if cached else ""
        timing_txt = f"detect {detect_ms:5.0f}ms  locate {locate_ms:5.2f}ms{suffix}"
        # Fixed-size background box (sized for the widest possible text,
        # including "cached") so position/size never jitters between frames
        # regardless of digit width or whether the cached suffix is shown.
        box_x, box_y, box_w, box_h = grid.shape[1] - 380, 8, 372, 34
        cv2.rectangle(grid, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
        cv2.putText(
            grid,
            timing_txt,
            (box_x + 8, box_y + box_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return vis, grid, logs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO ground-ref point -> light floor grid")
    p.add_argument("--source", default=str(DEFAULT_IMAGE))
    p.add_argument("--calib", default=str(DEFAULT_CALIB))
    p.add_argument("--model", default="yolo26s.pt", help="Ultralytics detect weights (e.g. yolo26n/s/m.pt)")
    p.add_argument("--conf", type=float, default=0.45, help="higher reduces desk/monitor FPs")
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
    p.add_argument("--min-h-ratio", type=float, default=0.12, help="min box height / frame height")
    p.add_argument("--min-aspect", type=float, default=1.15, help="min box height / width")
    p.add_argument(
        "--min-bottom-ratio",
        type=float,
        default=0.28,
        help="reject boxes whose bottom is above this frame-height ratio",
    )
    p.add_argument(
        "--valid-xmin",
        type=float,
        default=0.0,
        help="cells with X right-edge <= this are marked low-confidence/desk gray; "
        "0 disables desk zone (full grid). Use 170 to restore old desk mask.",
    )
    p.add_argument("--max-width", type=int, default=1280)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument(
        "--no-timing",
        action="store_true",
        help="hide detect/locate timing on the Grid window",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="run YOLO every N frames; skipped frames reuse last detections (default 1=every frame)",
    )
    p.add_argument(
        "--cell-hold",
        type=int,
        default=2,
        help="a cell only lights/clears after N consecutive DETECTION RUNS agree "
        "(counted in stride units, not raw frames); 1 disables debounce",
    )
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

    if args.stride < 1:
        raise SystemExit("--stride 必須 >= 1")

    cam_win = "Detect + Grid"
    grid_win = "Grid"
    print(f"偵測：YOLO（{args.model}），conf={args.conf}（每幀獨立，無 ID／追蹤）")
    print(f"參考點模式：{args.ref}（紅=foot，紫=head_drop）。按 q 結束，s 存圖。")
    print(f"預覽寬度固定 max-width={args.max_width}（影片與格子視窗皆鎖定畫面像素大小）")
    if args.stride > 1:
        print(f"跳幀：每 {args.stride} 幀才跑 YOLO，中間幀沿用上次偵測結果。")
    if args.cell_hold > 1:
        print(f"防抖：格子需連續 {args.cell_hold} 次偵測結果一致才會點亮／熄滅。")
    if not args.no_timing:
        print("計時：僅在偵測到人時顯示於格子上方（detect=辨識，locate=定位）。")

    det_kw = dict(
        conf=args.conf,
        ref=args.ref,
        aspect=args.aspect,
        truncate_ratio=args.truncate_ratio,
        min_h_ratio=args.min_h_ratio,
        min_aspect=args.min_aspect,
        min_bottom_ratio=args.min_bottom_ratio,
    )

    def process_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dets, detect_ms, locate_ms = detect_and_locate(frame, model, h_mat, **det_kw)
        timing = None if args.no_timing else (detect_ms, locate_ms)
        vis, grid, logs = render_detection_view(
            frame, dets, h_mat, args.valid_xmin, timing=timing, cached=False
        )
        for line in logs:
            print(line)
        return vis, grid

    if is_image:
        frame = imread_unicode(Path(source))
        if frame is None:
            raise SystemExit(f"無法讀取影像：{source}")
        vis, grid = process_frame(frame)
        view = resize_for_preview(vis, args.max_width)
        while True:
            show_fixed_window(cam_win, view)
            show_grid_window(grid_win, grid)
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

    use_latest = source.lower().startswith("rtsp://")
    reader: LatestFrameCapture | None = LatestFrameCapture(cap) if use_latest else None
    if use_latest:
        print("RTSP：啟用最新幀讀取（推論慢時丟棄舊幀，降低延遲感）")
        # wait briefly for first frame
        for _ in range(50):
            ok, frame = reader.read()
            if ok and frame is not None:
                break
            time.sleep(0.05)
        else:
            reader.release()
            raise SystemExit("RTSP 連線後未收到畫面。")

    stabilizer = CellStabilizer(args.cell_hold)
    confirmed_cells: set[tuple[int, int]] = set()

    try:
        frame_idx = 0
        last_dets: list[dict] = []
        last_timing: tuple[float, float] | None = None
        while True:
            if reader is not None:
                ok, frame = reader.read()
                if not ok or frame is None:
                    if not reader.is_alive():
                        print("讀取結束或失敗。")
                        break
                    time.sleep(0.01)
                    continue
            else:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("讀取結束或失敗。")
                    break

            frame_idx += 1
            run_detect = frame_idx == 1 or (frame_idx - 1) % args.stride == 0
            if run_detect:
                last_dets, detect_ms, locate_ms = detect_and_locate(
                    frame, model, h_mat, **det_kw
                )
                last_timing = None if args.no_timing else (detect_ms, locate_ms)
                raw_cells = {d["cell"] for d in last_dets if d.get("cell") is not None}
                confirmed_cells = stabilizer.update(raw_cells)
            timing = last_timing if not args.no_timing else None
            vis, grid, logs = render_detection_view(
                frame,
                last_dets,
                h_mat,
                args.valid_xmin,
                timing=timing,
                cached=not run_detect,
                grid_cells=confirmed_cells,
            )
            if run_detect:
                for line in logs:
                    print(line)
            view = resize_for_preview(vis, args.max_width)
            show_fixed_window(cam_win, view)
            show_grid_window(grid_win, grid)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                imwrite_unicode(Path(args.out), view)
                imwrite_unicode(Path(args.out).with_name("detect_grid_cells.jpg"), grid)
                print(f"已存：{args.out}")
            elif key in (ord("q"), 27):
                break
    finally:
        if reader is not None:
            reader.release()
        else:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

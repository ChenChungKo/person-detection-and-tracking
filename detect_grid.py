"""Detect person (YOLO) + ground ref + light grid cell.

Ground reference:
  - foot / head_drop / auto (see --ref)

Tracking (same person -> same ID):
  - default: Ultralytics ByteTrack (see --tracker)
  - disable with --no-track

Usage:
  python detect_grid.py --source test/test.mp4 --ref auto
  python detect_grid.py --source test/test.mp4 --stride 3 --no-track
  python detect_grid.py --source "rtsp://user:pass@ip:554/stream1" --ref auto
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
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
from stable_id import StableIdMapper

DEFAULT_IMAGE = Path(__file__).resolve().parent / "test" / "static_frame.jpg"
DEFAULT_CALIB = Path(__file__).resolve().parent / "calibration" / "homography.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "test" / "detect_grid_preview.jpg"
DEFAULT_TRACKER = Path(__file__).resolve().parent / "trackers" / "bytetrack_stable.yaml"


def format_id_list(ids: tuple[int, ...]) -> str:
    return ",".join(f"ID{i}" for i in ids) if ids else "—"


def log_id_change(
    frame_idx: int,
    fps: float,
    t0: float,
    prev: tuple[int, ...] | None,
    curr: tuple[int, ...],
) -> None:
    """Print when the set of active stable IDs changes."""
    wall = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    elapsed = time.perf_counter() - t0
    video_t = frame_idx / fps if fps > 1e-6 else 0.0
    prev_s = format_id_list(prev) if prev is not None else "(start)"
    curr_s = format_id_list(curr)
    print(
        f"[ID-CHANGE] {wall}  elapsed={elapsed:7.2f}s  "
        f"video={video_t:7.2f}s  frame={frame_idx:5d}  {prev_s} → {curr_s}",
        flush=True,
    )


def resize_for_preview(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if max_width <= 0 or w <= max_width:
        return frame.copy()
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
    """Return person ground-ref points from YOLO detect/track boxes."""
    out: list[dict] = []
    if result.boxes is None or len(result.boxes) == 0:
        return out
    boxes = result.boxes
    has_ids = boxes.id is not None
    for i in range(len(boxes)):
        if int(boxes.cls[i].item()) != 0:
            continue
        conf = float(boxes.conf[i].item())
        if conf < conf_thres:
            continue
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        track_id = int(boxes.id[i].item()) if has_ids else None
        # Same geometry gate for detect and track — tracked FPs (monitor/chair)
        # used to skip this and steal stable IDs.
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
                "track_id": track_id,
            }
        )
    return out


def scale_detections_for_preview(detections: list[dict], scale: float) -> list[dict]:
    """Scale box/foot drawing coords; keep world/cell in original units."""
    if abs(scale - 1.0) < 1e-6:
        return detections
    out: list[dict] = []
    for det in detections:
        d = dict(det)
        x1, y1, x2, y2 = det["xyxy"]
        d["xyxy"] = (
            int(round(x1 * scale)),
            int(round(y1 * scale)),
            int(round(x2 * scale)),
            int(round(y2 * scale)),
        )
        fx, fy = det["foot"]
        d["foot"] = (fx * scale, fy * scale)
        if "head" in det:
            hx, hy = det["head"]
            d["head"] = (hx * scale, hy * scale)
        out.append(d)
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
    out_margin: float = 45.0,
) -> tuple[np.ndarray, set[tuple[int, int]], list[str]]:
    vis = frame  # caller passes a writable preview-sized copy
    cells: set[tuple[int, int]] = set()
    logs: list[str] = []
    fs = max(0.75, frame.shape[1] / 1280.0)
    thick = max(2, int(round(2 * fs)))

    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        fx, fy = det["foot"]
        used = det.get("mode", "foot")
        tid = det.get("track_id")
        id_txt = f"ID{tid}" if tid is not None else "person"
        box_color = (0, 255, 0)
        if "world" in det and "cell" in det:
            wx, wy = det["world"]
            cell = det["cell"]
        else:
            wx, wy = image_to_world(h_mat, fx, fy)
            cell = world_to_cell(wx, wy, margin_cm=out_margin)

        cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, max(2, thick))
        ref_color = (0, 0, 255) if used == "foot" else (255, 0, 255)
        cv2.circle(vis, (int(fx), int(fy)), max(6, int(5 * fs)), ref_color, -1)

        label_y = y1 - 12
        if label_y < int(40 * fs):
            label_y = y1 + int(36 * fs)

        if cell is None:
            put_label(
                vis,
                f"{id_txt} OUT",
                (x1, label_y),
                fg=(255, 255, 255),
                bg=(0, 0, 220),
                scale=0.9 * fs,
                thickness=thick,
            )
            logs.append(
                f"{id_txt} conf={det['conf']:.2f} {used}=({fx:.1f},{fy:.1f}) "
                f"world=({wx:.1f},{wy:.1f}) OUT"
            )
        else:
            cells.add(cell)
            low_conf = used == "head_drop"
            if valid_xmin > 0 and X_EDGES[cell[0] + 1] <= valid_xmin:
                low_conf = True
            cell_txt = cell_label(*cell)
            put_label(
                vis,
                id_txt,
                (x1, label_y),
                fg=(0, 0, 0),
                bg=(0, 255, 255),
                scale=0.85 * fs,
                thickness=thick,
            )
            logs.append(
                f"{id_txt} conf={det['conf']:.2f} "
                f"world=({wx:.1f},{wy:.1f}) {cell_txt}"
                + (" [low]" if low_conf else "")
            )

    return vis, cells, logs


class GridCache:
    """Avoid redrawing the floor grid when occupied cells are unchanged."""

    def __init__(self) -> None:
        self._cells: set[tuple[int, int]] | None = None
        self._valid_xmin: float | None = None
        self._base: np.ndarray | None = None

    def get(self, cells: set[tuple[int, int]], valid_xmin: float) -> np.ndarray:
        if (
            self._base is not None
            and self._cells == cells
            and self._valid_xmin == valid_xmin
        ):
            return self._base.copy()
        img = draw_multi_grid(cells, valid_xmin)
        self._cells = set(cells)
        self._valid_xmin = valid_xmin
        self._base = img
        return img.copy()


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
    track: bool = True,
    tracker: str | None = None,
    imgsz: int = 640,
    out_margin: float = 45.0,
) -> tuple[list[dict], float, float]:
    t0 = time.perf_counter()
    infer_kw = dict(conf=conf, classes=[0], imgsz=imgsz, verbose=False)
    if track:
        results = model.track(
            frame,
            persist=True,
            tracker=tracker or str(DEFAULT_TRACKER),
            **infer_kw,
        )
    else:
        results = model.predict(frame, **infer_kw)
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
        det["cell"] = world_to_cell(wx, wy, margin_cm=out_margin)
    # Drop floating FPs: box sits mid-frame but Homography shoots far outside.
    frame_h = float(frame.shape[0])
    cleaned: list[dict] = []
    for det in dets:
        _x1, _y1, _x2, y2 = det["xyxy"]
        cell = det.get("cell")
        wx, wy = det["world"]
        far_out = cell is None and (
            wy < -out_margin * 2
            or wx < -out_margin * 2
            or wx > 530.0 + out_margin * 2
            or wy > 540.0 + out_margin * 2
        )
        elevated = y2 < 0.50 * frame_h
        if far_out and elevated:
            continue
        cleaned.append(det)
    dets = cleaned
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
    max_width: int = 1280,
    grid_cache: GridCache | None = None,
    out_margin: float = 45.0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    # Draw on preview-sized frame (much cheaper than annotating 2880px then resize).
    view = resize_for_preview(frame, max_width)
    scale = view.shape[1] / float(frame.shape[1]) if frame.shape[1] else 1.0
    draw_dets = scale_detections_for_preview(dets, scale)
    vis, cells, logs = annotate_and_cells(
        view, draw_dets, h_mat, valid_xmin, out_margin=out_margin
    )
    display_cells = grid_cells if grid_cells is not None else cells
    if grid_cache is not None:
        grid = grid_cache.get(display_cells, valid_xmin)
    else:
        grid = draw_multi_grid(display_cells, valid_xmin)

    if timing is not None:
        detect_ms, locate_ms = timing
        suffix = "  cached" if cached else ""
        timing_txt = f"detect {detect_ms:5.0f}ms  locate {locate_ms:5.2f}ms{suffix}"
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
    p.add_argument("--conf", type=float, default=0.50, help="higher reduces desk/monitor FPs")
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
    p.add_argument("--min-h-ratio", type=float, default=0.14, help="min box height / frame height")
    p.add_argument("--min-aspect", type=float, default=1.25, help="min box height / width")
    p.add_argument(
        "--min-bottom-ratio",
        type=float,
        default=0.35,
        help="reject boxes whose bottom is above this frame-height ratio (monitors)",
    )
    p.add_argument(
        "--valid-xmin",
        type=float,
        default=0.0,
        help="cells with X right-edge <= this are marked low-confidence/desk gray; "
        "0 disables desk zone (full grid). Use 170 to restore old desk mask.",
    )
    p.add_argument(
        "--out-margin",
        type=float,
        default=45.0,
        help="cm: if foot world point is only this far outside the grid, "
        "snap into the nearest edge cell instead of marking OUT (default 45=1 tile)",
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
        default=2,
        help="run YOLO every N frames; skipped frames reuse last detections "
        "(default 2; use 1 only if you need denser track updates)",
    )
    p.add_argument(
        "--cell-hold",
        type=int,
        default=2,
        help="a cell only lights/clears after N consecutive DETECTION RUNS agree "
        "(counted in stride units, not raw frames); 1 disables debounce",
    )
    p.add_argument(
        "--track",
        dest="track",
        action="store_true",
        default=True,
        help="enable ByteTrack IDs via model.track (default on for video)",
    )
    p.add_argument(
        "--no-track",
        dest="track",
        action="store_false",
        help="disable tracking; each frame is independent detect",
    )
    p.add_argument(
        "--tracker",
        default=str(DEFAULT_TRACKER),
        help="Ultralytics tracker yaml (default: trackers/bytetrack_stable.yaml)",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference size (default 640; higher is slower)",
    )
    p.add_argument(
        "--realtime",
        dest="realtime",
        action="store_true",
        default=True,
        help="for local video: skip frames to keep near realtime (default on)",
    )
    p.add_argument(
        "--no-realtime",
        dest="realtime",
        action="store_false",
        help="process every frame even if playback becomes slow-motion",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="do not print per-detection lines (ID change log still prints unless --no-log-id)",
    )
    p.add_argument(
        "--log-id",
        dest="log_id",
        action="store_true",
        default=True,
        help="print ID set changes with timestamps (default on)",
    )
    p.add_argument(
        "--no-log-id",
        dest="log_id",
        action="store_false",
        help="disable ID-change timestamp logs",
    )
    p.add_argument(
        "--single-person",
        dest="single_person",
        action="store_true",
        default=False,
        help="DEMO ONLY: assume one person and force-reuse their ID (default off)",
    )
    p.add_argument(
        "--id-max-dist",
        type=float,
        default=400.0,
        help="cap on floor rematch distance (cm)",
    )
    p.add_argument(
        "--id-max-gap",
        type=int,
        default=400,
        help="max frames before dropping pending raw-ID hit counters",
    )
    p.add_argument(
        "--id-max-speed",
        type=float,
        default=200.0,
        help="max walk speed (cm/s) used to limit rematch distance by time gap",
    )
    p.add_argument(
        "--appear-thresh",
        type=float,
        default=0.40,
        help="Re-ID cosine thresh to reuse an existing ID (higher = stricter; default 0.40)",
    )
    p.add_argument(
        "--min-hits",
        type=int,
        default=3,
        help="detect-runs that must FAIL gallery before minting a NEW ID "
        "(default 3; with --stride 2 ≈ 0.3s at 20fps)",
    )
    p.add_argument(
        "--id-coast",
        type=int,
        default=60,
        help="keep last IDs for this many frames when detection briefly drops (default 60)",
    )
    p.add_argument(
        "--id-sticky",
        type=int,
        default=45,
        help="short-gap motion fallback only when Re-ID is weak (default 45)",
    )
    p.add_argument(
        "--max-prototypes",
        type=int,
        default=0,
        help="max appearance looks per ID (0 = unlimited; default 0). "
        "New looks are added only when appearance shifts a lot",
    )
    p.add_argument(
        "--gallery-dir",
        default=str(Path(__file__).resolve().parent / "test" / "reid_gallery"),
        help="save per-ID appearance crop images here (cleared each run)",
    )
    p.add_argument(
        "--no-gallery-dump",
        action="store_true",
        help="do not save appearance gallery crop images",
    )
    p.add_argument(
        "--reid",
        dest="reid",
        action="store_true",
        default=True,
        help="use YOLO26 Re-ID embeddings for appearance (default on)",
    )
    p.add_argument(
        "--no-reid",
        dest="reid",
        action="store_false",
        help="fall back to HSV clothing histogram instead of Re-ID",
    )
    p.add_argument(
        "--reid-model",
        default="yolo26n-reid.onnx",
        help="Ultralytics Re-ID model (e.g. yolo26n-reid.onnx / yolo26s-reid.onnx)",
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

    tracker_path = Path(args.tracker)
    if args.track and not tracker_path.exists():
        raise SystemExit(f"找不到 tracker 設定：{tracker_path}")

    cam_win = "Detect + Grid"
    grid_win = "Grid"
    if args.track:
        print(
            f"偵測＋追蹤：YOLO（{args.model}）+ ByteTrack（{tracker_path.name}），"
            f"conf={args.conf}，imgsz={args.imgsz}，stride={args.stride}"
        )
    else:
        print(
            f"偵測：YOLO（{args.model}），conf={args.conf}，"
            f"imgsz={args.imgsz}（每幀獨立，無 ID／追蹤）"
        )
    print(f"參考點模式：{args.ref}（紅=foot，紫=head_drop）。按 q 結束，s 存圖。")
    print(f"預覽寬度固定 max-width={args.max_width}（影片與格子視窗皆鎖定畫面像素大小）")
    if args.stride > 1:
        print(f"跳幀：每 {args.stride} 幀才跑 YOLO，中間幀沿用上次偵測／ID。")
    if args.cell_hold > 1:
        print(f"防抖：格子需連續 {args.cell_hold} 次偵測結果一致才會點亮／熄滅。")
    if args.out_margin > 0:
        print(
            f"OUT 容差：世界座標超出格子 ≤ {args.out_margin:g} cm 時夾回邊緣格"
            "（--out-margin 0 關閉）。"
        )
    if not args.no_timing:
        print("計時：顯示於格子上方（detect=辨識，locate=定位）。")

    det_kw = dict(
        conf=args.conf,
        ref=args.ref,
        aspect=args.aspect,
        truncate_ratio=args.truncate_ratio,
        min_h_ratio=args.min_h_ratio,
        min_aspect=args.min_aspect,
        min_bottom_ratio=args.min_bottom_ratio,
        track=args.track and not is_image,
        tracker=str(tracker_path),
        imgsz=args.imgsz,
        out_margin=args.out_margin,
    )
    grid_cache = GridCache()

    def process_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dets, detect_ms, locate_ms = detect_and_locate(frame, model, h_mat, **det_kw)
        timing = None if args.no_timing else (detect_ms, locate_ms)
        vis, grid, logs = render_detection_view(
            frame,
            dets,
            h_mat,
            args.valid_xmin,
            timing=timing,
            cached=False,
            max_width=args.max_width,
            grid_cache=grid_cache,
            out_margin=args.out_margin,
        )
        if not args.quiet:
            for line in logs:
                print(line)
        return vis, grid

    if is_image:
        frame = imread_unicode(Path(source))
        if frame is None:
            raise SystemExit(f"無法讀取影像：{source}")
        vis, grid = process_frame(frame)
        while True:
            if not show_fixed_window(cam_win, vis) or not show_grid_window(grid_win, grid):
                break
            key = cv2.waitKey(20) & 0xFF
            if key == ord("s"):
                imwrite_unicode(Path(args.out), vis)
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
    is_file_video = (not use_latest) and Path(source).suffix.lower() in {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
    }
    reader: LatestFrameCapture | None = LatestFrameCapture(cap) if use_latest else None
    if use_latest:
        print("RTSP：啟用最新幀讀取（推論慢時丟棄舊幀，降低延遲感）")
        for _ in range(50):
            ok, frame = reader.read()
            if ok and frame is not None:
                break
            time.sleep(0.05)
        else:
            reader.release()
            raise SystemExit("RTSP 連線後未收到畫面。")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if video_fps <= 1e-3:
        video_fps = 20.0
    use_realtime = bool(args.realtime and is_file_video)
    if use_realtime:
        print(
            f"本機影片：即時跟播（約 {video_fps:.1f} fps）；"
            "推論慢會自動丟幀。完整逐幀請加 --no-realtime"
        )

    stabilizer = CellStabilizer(args.cell_hold)
    confirmed_cells: set[tuple[int, int]] = set()

    reid_encoder = None
    if args.track and args.reid:
        try:
            from reid_encoder import PersonReIDEncoder, resolve_reid_model

            model_name = resolve_reid_model(args.reid_model)
            print(f"載入 Re-ID：{model_name} …")
            reid_encoder = PersonReIDEncoder(model_name)
            print("Re-ID 就緒（外貌比對改用深度特徵，不再用 HSV 顏色）。")
        except Exception as exc:  # noqa: BLE001
            print(f"Re-ID 載入失敗，改回 HSV：{exc}")
            reid_encoder = None

    id_mapper = StableIdMapper(
        max_dist_cm=args.id_max_dist,
        max_gap_frames=args.id_max_gap,
        max_speed_cm_s=args.id_max_speed,
        appear_thresh=args.appear_thresh,
        fps=video_fps if is_file_video or use_latest else 20.0,
        single_person=args.single_person,
        min_hits=args.min_hits,
        encoder=reid_encoder,
        coast_frames=args.id_coast,
        sticky_frames=args.id_sticky,
        max_prototypes=args.max_prototypes,
        gallery_dir=None if args.no_gallery_dump else args.gallery_dir,
    )
    if args.track:
        if args.single_person:
            print("ID 穩定層：單人強制接回（demo，非真實辨識）")
        else:
            appear_mode = "Re-ID" if reid_encoder is not None else "HSV"
            proto_s = (
                "不限外貌原型數"
                if args.max_prototypes <= 0
                else f"每人最多 {args.max_prototypes} 種外貌原型"
            )
            print(
                f"ID 穩定層：{appear_mode} 圖庫為主（離開多久皆可接回）；"
                f"appear≥{args.appear_thresh:.2f}，{proto_s}（換裝可並存）；"
                f"首次發 ID 立即存 first.jpg；"
                f"新 ID 需連續 {args.min_hits} 次偵測對不上圖庫才發號；"
                f"同分時優先較舊 ID。"
            )
        if not args.no_gallery_dump:
            print(f"外貌圖庫裁切圖：{args.gallery_dir}（每次重跑會清空重存）")
        print(
            f"誤檢過濾：conf≥{args.conf}，min_bottom={args.min_bottom_ratio}，"
            f"min_aspect={args.min_aspect}，min_h={args.min_h_ratio}"
        )
        if args.log_id:
            print("ID 變化會印 [ID-CHANGE]（時刻 / elapsed / video / frame）。")

    try:
        frame_idx = 0
        last_dets: list[dict] = []
        last_timing: tuple[float, float] | None = None
        last_id_key: tuple[int, ...] | None = None
        t_play0 = time.perf_counter()
        log_fps = video_fps if (is_file_video or use_latest) else 20.0
        while True:
            if reader is not None:
                ok, frame = reader.read()
                if not ok or frame is None:
                    if not reader.is_alive():
                        print("讀取結束或失敗。")
                        break
                    time.sleep(0.01)
                    continue
                frame_idx += 1
            else:
                if use_realtime:
                    # Drop frames so wall-clock stays near video timeline.
                    target = int((time.perf_counter() - t_play0) * video_fps) + 1
                    while frame_idx + 1 < target:
                        if not cap.grab():
                            ok, frame = False, None
                            break
                        frame_idx += 1
                    else:
                        ok, frame = cap.read()
                        if ok:
                            frame_idx += 1
                else:
                    ok, frame = cap.read()
                    if ok:
                        frame_idx += 1
                if not ok or frame is None:
                    print("讀取結束或失敗。")
                    break

            run_detect = frame_idx == 1 or (frame_idx - 1) % args.stride == 0
            if run_detect:
                last_dets, detect_ms, locate_ms = detect_and_locate(
                    frame, model, h_mat, **det_kw
                )
                if det_kw.get("track"):
                    last_dets = id_mapper.apply(last_dets, frame_idx, frame=frame)
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
                max_width=args.max_width,
                grid_cache=grid_cache,
                out_margin=args.out_margin,
            )
            if run_detect and det_kw.get("track") and args.log_id:
                id_key = tuple(
                    sorted(d["track_id"] for d in last_dets if d.get("track_id") is not None)
                )
                if id_key != last_id_key:
                    log_id_change(frame_idx, log_fps, t_play0, last_id_key, id_key)
                    last_id_key = id_key
            if run_detect and not args.quiet:
                for line in logs:
                    print(line)
            if not show_fixed_window(cam_win, vis) or not show_grid_window(grid_win, grid):
                break

            if use_realtime:
                # If we are ahead of the timeline, wait a bit.
                ahead = frame_idx / video_fps - (time.perf_counter() - t_play0)
                wait_ms = 1
                if ahead > 0.005:
                    wait_ms = max(1, int(ahead * 1000))
                key = cv2.waitKey(wait_ms) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                imwrite_unicode(Path(args.out), vis)
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

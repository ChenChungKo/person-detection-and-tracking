"""Detect person (YOLO) + ground ref + light grid cell.

Ground reference:
  - foot / head_drop / auto (see --ref)

Tracking (Ultralytics docs: https://docs.ultralytics.com/modes/track):
  - default: YOLO.track persist=True + BoT-SORT (trackers/botsort.yaml)
  - long-term IDs: Stable-ID + OSNet gallery (not the tracker yaml)
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
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from grid_occupancy import (
    FLOOR_MARKS,
    X_EDGES,
    cell_label,
    draw_grid,
    id_bgr_color,
    imread_unicode,
    imwrite_unicode,
    landmark_bgr,
    show_fixed_window,
    show_grid_window,
    world_to_cell,
)
from latest_frame import LatestFrameCapture
from stable_id import StableIdMapper

DEFAULT_IMAGE = Path(__file__).resolve().parent / "test" / "static_frame.jpg"
DEFAULT_CALIB = Path(__file__).resolve().parent / "calibration" / "homography.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "test" / "detect_grid_preview.jpg"
DEFAULT_TRACKER = Path(__file__).resolve().parent / "trackers" / "botsort.yaml"


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


def stack_demo_views(camera: np.ndarray, grid: np.ndarray, height: int = 720) -> np.ndarray:
    """Place camera and floor-grid views side by side for video export."""

    def fit_height(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        width = max(1, int(round(w * height / float(h))))
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    left = fit_height(camera)
    right = fit_height(grid)
    separator = np.full((height, 2, 3), (40, 40, 40), dtype=np.uint8)
    return np.hstack((left, separator, right))


def load_homography(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return np.array(payload["homography"], dtype=np.float64)


def image_to_world(h_mat: np.ndarray, x: float, y: float) -> tuple[float, float]:
    pts = np.array([[[x, y]]], dtype=np.float64)
    world = cv2.perspectiveTransform(pts, h_mat)[0, 0]
    return float(world[0]), float(world[1])


def world_to_image(h_inv: np.ndarray, x: float, y: float) -> tuple[float, float]:
    pts = np.array([[[x, y]]], dtype=np.float64)
    pix = cv2.perspectiveTransform(pts, h_inv)[0, 0]
    return float(pix[0]), float(pix[1])


class FloorOverlayCache:
    """Homography is static: project the five floor marks once per preview size."""

    def __init__(self) -> None:
        self._key: tuple[int, int, int, int] | None = None
        self._h_inv: np.ndarray | None = None
        self._marks: list[tuple[str, int, int, tuple[int, int, int]]] = []

    def prepare(self, h_mat: np.ndarray, frame_wh: tuple[int, int], preview_wh: tuple[int, int]) -> None:
        fw, fh = frame_wh
        pw, ph = preview_wh
        key = (fw, fh, pw, ph)
        if self._key == key and self._h_inv is not None:
            return
        self._key = key
        self._h_inv = np.linalg.inv(h_mat)
        scale = pw / float(fw) if fw else 1.0
        marks: list[tuple[str, int, int, tuple[int, int, int]]] = []
        for name, wx, wy, rgb in FLOOR_MARKS:
            ix, iy = world_to_image(self._h_inv, wx, wy)
            px, py = int(round(ix * scale)), int(round(iy * scale))
            if -40 <= px < pw + 40 and -40 <= py < ph + 40:
                marks.append((name, px, py, landmark_bgr(rgb)))
        self._marks = marks

    def draw(self, vis: np.ndarray) -> None:
        for name, px, py, bgr in self._marks:
            cv2.circle(vis, (px, py), 12, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(vis, (px, py), 9, bgr, -1, cv2.LINE_AA)
            cv2.putText(
                vis,
                name,
                (px + 14, py + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                name,
                (px + 14, py + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                bgr,
                2,
                cv2.LINE_AA,
            )


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

    - foot: bbox bottom-center (true when ankles visible; also best for seated)
    - head_drop: from head drop by estimated *standing* height — only when the
      person is cut off by the bottom of the frame
    - auto: head_drop only if box looks truncated **and** touches frame bottom;
      mid-frame short boxes (typical sitting) keep foot = bbox bottom, otherwise
      Homography shoots into the aisle behind the chair
    """
    cx = 0.5 * (x1 + x2)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    head_x, head_y = cx, float(y1)
    foot_x, foot_y = cx, float(y2)
    y_max = float(frame_h - 1)

    looks_truncated = (bh / bw) < truncate_ratio
    # Standing person cut by bottom edge — not a seated torso floating mid-frame.
    cut_by_bottom = y2 >= 0.90 * frame_h

    if mode == "foot":
        return foot_x, min(foot_y, y_max), "foot"
    if mode == "head_drop":
        est_h = bw * aspect
        est_foot_y = head_y + est_h
        return head_x, min(est_foot_y, y_max), "head_drop"

    # auto: never extrapolate standing height for seated / mid-frame boxes.
    if looks_truncated and cut_by_bottom:
        est_h = bw * aspect
        est_foot_y = max(foot_y, head_y + est_h)
        return head_x, min(est_foot_y, y_max), "head_drop"
    return foot_x, min(foot_y, y_max), "foot"


def is_plausible_person_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_h: int,
    frame_w: int,
    min_h_ratio: float = 0.06,
    min_aspect: float = 0.8,
    min_bottom_ratio: float = 0.12,
    max_aspect: float = 4.5,
) -> bool:
    """Drop only obvious non-person fragments (hands / top-of-frame monitors).

    Seated classmates are often short/wide; a strict full-body gate hid them.
    """
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    if bh < min_h_ratio * frame_h:
        return False
    if bh < 80:
        return False
    if bw < max(40, 0.02 * frame_w):
        return False
    aspect = bh / bw
    if aspect < min_aspect:
        return False
    if aspect > max_aspect:
        return False
    if y2 < min_bottom_ratio * frame_h:
        return False
    if (bw * bh) < 0.004 * frame_w * frame_h:
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
    min_h_ratio: float = 0.06,
    min_aspect: float = 0.8,
    min_bottom_ratio: float = 0.12,
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
        track_id = None
        if has_ids:
            raw = float(boxes.id[i].item())
            if raw == raw:
                track_id = int(raw)
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


class DetectionCoaster:
    """Extrapolate boxes on skipped frames so overlays keep up with motion."""

    def __init__(self) -> None:
        self._hist: dict[int, list[tuple[int, float, float, float, float]]] = {}

    def observe(self, dets: list[dict], frame_idx: int) -> None:
        seen: set[int] = set()
        for det in dets:
            tid = det.get("track_id")
            if tid is None:
                continue
            tid = int(tid)
            seen.add(tid)
            x1, y1, x2, y2 = det["xyxy"]
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            hist = self._hist.setdefault(tid, [])
            hist.append((frame_idx, cx, cy, float(x2 - x1), float(y2 - y1)))
            del hist[:-3]
        for tid in list(self._hist):
            if tid not in seen:
                self._hist.pop(tid, None)

    def extrapolate(self, dets: list[dict], frame_idx: int) -> list[dict]:
        out: list[dict] = []
        for det in dets:
            d = dict(det)
            tid = d.get("track_id")
            if tid is None:
                out.append(d)
                continue
            hist = self._hist.get(int(tid), [])
            if len(hist) < 2:
                out.append(d)
                continue
            (_f0, cx0, cy0, _w0, _h0) = hist[-2]
            (f1, cx1, cy1, w1, h1) = hist[-1]
            dt = max(1, f1 - hist[-2][0])
            steps = frame_idx - f1
            if steps <= 0:
                out.append(d)
                continue
            steps = min(int(steps), 6)
            vcx = (cx1 - cx0) / dt
            vcy = (cy1 - cy0) / dt
            cx = cx1 + vcx * steps
            cy = cy1 + vcy * steps
            x1 = int(round(cx - 0.5 * w1))
            y1 = int(round(cy - 0.5 * h1))
            x2 = int(round(cx + 0.5 * w1))
            y2 = int(round(cy + 0.5 * h1))
            d["xyxy"] = (x1, y1, x2, y2)
            if "foot" in d:
                fx, fy = d["foot"]
                d["foot"] = (fx + vcx * steps, fy + vcy * steps)
            if "head" in d:
                hx, hy = d["head"]
                d["head"] = (hx + vcx * steps, hy + vcy * steps)
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
    show_cell_label: bool = False,
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
        box_color = id_bgr_color(tid) if tid is not None else (0, 255, 0)
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
            tag = f"{id_txt} ({cell[0]},{cell[1]})" if show_cell_label else id_txt
            put_label(
                vis,
                tag,
                (x1, label_y),
                fg=(0, 0, 0),
                bg=box_color,
                scale=0.85 * fs,
                thickness=thick,
            )
            logs.append(
                f"{id_txt} conf={det['conf']:.2f} "
                f"world=({wx:.1f},{wy:.1f}) {cell_txt}"
                + (" [low]" if low_conf else "")
            )

    return vis, cells, logs


def occupancy_from_dets(
    dets: list[dict],
    allowed_cells: set[tuple[int, int]] | None = None,
) -> dict[tuple[int, int], list[int]]:
    """Map floor cell -> stable track IDs currently standing there."""
    occ: dict[tuple[int, int], list[int]] = {}
    for det in dets:
        cell = det.get("cell")
        tid = det.get("track_id")
        if cell is None or tid is None:
            continue
        cell_t = (int(cell[0]), int(cell[1]))
        if allowed_cells is not None and cell_t not in allowed_cells:
            continue
        occ.setdefault(cell_t, []).append(int(tid))
    for cell, ids in occ.items():
        occ[cell] = sorted(set(ids))
    return occ


def occupancy_key(
    occ: dict[tuple[int, int], list[int]],
) -> frozenset[tuple[tuple[int, int], frozenset[int]]]:
    return frozenset((cell, frozenset(ids)) for cell, ids in occ.items())


class GridCache:
    """Avoid redrawing the floor grid when occupancy / IDs are unchanged."""

    def __init__(self) -> None:
        self._key: frozenset[tuple[tuple[int, int], frozenset[int]]] | None = None
        self._valid_xmin: float | None = None
        self._landmarks = False
        self._base: np.ndarray | None = None

    def get(
        self,
        occupancy: dict[tuple[int, int], list[int]],
        valid_xmin: float,
        landmarks: bool = False,
    ) -> np.ndarray:
        key = occupancy_key(occupancy)
        if (
            self._base is not None
            and self._key == key
            and self._valid_xmin == valid_xmin
            and self._landmarks == landmarks
        ):
            return self._base.copy()
        img = draw_occupancy_grid(occupancy, valid_xmin, landmarks=landmarks)
        self._key = key
        self._valid_xmin = valid_xmin
        self._landmarks = landmarks
        self._base = img
        return img.copy()


def draw_occupancy_grid(
    occupancy: dict[tuple[int, int], list[int]],
    valid_xmin: float,
    landmarks: bool = False,
) -> np.ndarray:
    """Light occupied cells and label each with person ID(s)."""
    return draw_grid(
        None, valid_x_min=valid_xmin, occupancy=occupancy, landmarks=landmarks
    )


def draw_multi_grid(cells: set[tuple[int, int]], valid_xmin: float) -> np.ndarray:
    """Light all occupied cells (no ID labels; kept for simple callers)."""
    occ = {cell: [] for cell in cells}
    # Empty ID list still lights via active-style path — use dummy? draw_grid
    # only colors by occupancy when ids non-empty. Fall back to old merge.
    if not cells:
        return draw_grid(None, valid_x_min=valid_xmin)
    base = draw_grid(None, valid_x_min=valid_xmin)
    for cell in cells:
        lit = draw_grid(cell, valid_x_min=valid_xmin)
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


class AsyncTrackWorker:
    """YOLO.track(persist=True) on a side thread so the preview stays live.

    Official docs want consecutive persist calls; we queue at most one frame so
    the tracker still sees an ordered stream without blocking cv2.imshow.
    """

    def __init__(self, model: YOLO, h_mat: np.ndarray, det_kw: dict, id_mapper) -> None:
        self._model = model
        self._h_mat = h_mat
        self._det_kw = det_kw
        self._id_mapper = id_mapper
        self._lock = threading.Lock()
        self._pending: tuple[np.ndarray, int] | None = None
        self._dets: list[dict] = []
        self._timing: tuple[float, float] | None = None
        self._idx = 0
        self._stop = False
        self._thread = threading.Thread(
            target=self._loop, name="yolo-track", daemon=True
        )
        self._thread.start()

    def try_submit(self, frame: np.ndarray, frame_idx: int) -> None:
        with self._lock:
            if self._pending is not None:
                return
            self._pending = (frame.copy(), frame_idx)

    def snapshot(self) -> tuple[list[dict], tuple[float, float] | None, int]:
        with self._lock:
            return [dict(d) for d in self._dets], self._timing, self._idx

    def stop(self) -> None:
        self._stop = True
        self._thread.join(timeout=3.0)

    def _loop(self) -> None:
        while not self._stop:
            with self._lock:
                job = self._pending
                self._pending = None
            if job is None:
                time.sleep(0.003)
                continue
            frame, frame_idx = job
            dets, detect_ms, locate_ms = detect_and_locate(
                frame, self._model, self._h_mat, **self._det_kw
            )
            if self._det_kw.get("track") and self._id_mapper is not None:
                dets = self._id_mapper.apply(dets, frame_idx, frame=frame)
            with self._lock:
                self._dets = dets
                self._timing = (detect_ms, locate_ms)
                self._idx = frame_idx


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
    grid_occupancy: dict[tuple[int, int], list[int]] | None = None,
    show_floor_grid: bool = False,
    floor_overlay: FloorOverlayCache | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    # Draw on preview-sized frame (much cheaper than annotating 2880px then resize).
    view = resize_for_preview(frame, max_width)
    scale = view.shape[1] / float(frame.shape[1]) if frame.shape[1] else 1.0
    if show_floor_grid:
        if floor_overlay is None:
            floor_overlay = FloorOverlayCache()
        floor_overlay.prepare(
            h_mat,
            (frame.shape[1], frame.shape[0]),
            (view.shape[1], view.shape[0]),
        )
        floor_overlay.draw(view)
    draw_dets = scale_detections_for_preview(dets, scale)
    vis, cells, logs = annotate_and_cells(
        view,
        draw_dets,
        h_mat,
        valid_xmin,
        out_margin=out_margin,
        show_cell_label=show_floor_grid,
    )
    display_cells = grid_cells if grid_cells is not None else cells
    if grid_occupancy is not None:
        occupancy = dict(grid_occupancy)
    else:
        # Only light cells that already have a stable ID → ID color from frame 1
        # (no empty-occupancy yellow flash before ID assignment).
        occupancy = occupancy_from_dets(dets, allowed_cells=display_cells)

    if grid_cache is not None:
        grid = grid_cache.get(occupancy, valid_xmin, landmarks=show_floor_grid)
    else:
        grid = draw_occupancy_grid(occupancy, valid_xmin, landmarks=show_floor_grid)

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
    p.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="YOLO confidence (default 0.45; lower to 0.1 keeps more seated/far people, "
        "higher drops monitors and door-edge fragments)",
    )
    p.add_argument(
        "--ref",
        choices=["auto", "foot", "head_drop"],
        default="auto",
        help="auto: foot by default; head_drop only if truncated AND cut by frame bottom "
        "(sitting uses bbox bottom — avoids aisle foot drift)",
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
    p.add_argument(
        "--min-h-ratio",
        type=float,
        default=0.06,
        help="min box height / frame height; only drops tiny hand FPs",
    )
    p.add_argument(
        "--min-aspect",
        type=float,
        default=0.8,
        help="min box height / width (seated people are often wider than 1.2)",
    )
    p.add_argument(
        "--min-bottom-ratio",
        type=float,
        default=0.12,
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
    p.add_argument(
        "--show-floor-grid",
        dest="show_floor_grid",
        action="store_true",
        default=True,
        help="overlay five floor marks (A–D + center O) on camera and bird-eye views",
    )
    p.add_argument(
        "--no-floor-grid",
        dest="show_floor_grid",
        action="store_false",
        help="hide floor-grid overlay and cell coordinates on person labels",
    )
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument(
        "--save-video",
        default=None,
        help="save camera + grid preview as an MP4 using this exact tracking run",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="do not open preview windows (useful with --save-video)",
    )
    p.add_argument(
        "--no-timing",
        action="store_true",
        help="hide detect/locate timing on the Grid window",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=5,
        help="YOLO+grid occupancy every N display frames (default 5). "
        "Camera still draws every frame with coasted boxes; "
        "higher stride makes floor-grid lights change less often (more continuous)",
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
        help="enable Ultralytics model.track persist=True (default on for video)",
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
        help="Ultralytics tracker yaml (default: trackers/botsort.yaml, docs default)",
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
        help="for local video: cap playback near source fps without dropping track frames",
    )
    p.add_argument(
        "--no-realtime",
        dest="realtime",
        action="store_false",
        help="do not pace local playback (fixed tracking frames are unchanged)",
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
        default=0.34,
        help="Re-ID cosine thresh to reuse an existing ID (default 0.34; higher = stricter)",
    )
    p.add_argument(
        "--min-hits",
        type=int,
        default=16,
        help="detect-runs that must FAIL gallery before minting a NEW ID "
        "(default 16; with --stride 5 ≈ 4s at 20fps — reduces ID6/ID7 splits)",
    )
    p.add_argument(
        "--id-coast",
        type=int,
        default=8,
        help="briefly keep last INTERIOR boxes when ALL detections drop "
        "(default 8 frames ≈ 0.4s; not multiplied by stride). "
        "Boxes at the image edge are cleared immediately (person left). "
        "ID rematch after a real leave uses --id-sticky / gallery, not this",
    )
    p.add_argument(
        "--id-sticky",
        type=int,
        default=100,
        help="short-gap occlusion recovery window in frames (default 100 ≈ 5s at 20fps)",
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
        "--review-dir",
        default=str(Path(__file__).resolve().parent / "test" / "reid_review"),
        help="per-run review crops for later cleanup (not used for live Re-ID)",
    )
    p.add_argument(
        "--review-every",
        type=int,
        default=10,
        help="save one review crop per ID every N video frames (default 10 ≈ 0.5s)",
    )
    p.add_argument(
        "--review-dump",
        action="store_true",
        help="save a review photo database (off by default so tracking stays fast)",
    )
    p.add_argument(
        "--no-review-dump",
        action="store_true",
        help="do not save the review photo database (default)",
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
        default="osnet_ain",
        help="Stable-ID appearance: osnet_ain / osnet / yolo26n-reid.onnx "
        "(BoT-SORT short ReID is set in trackers/botsort.yaml)",
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
            f"偵測＋追蹤：YOLO（{args.model}）+ BoT-SORT（{tracker_path.name}）"
            f" persist=True，conf={args.conf}，imgsz={args.imgsz}，stride={args.stride}"
        )
    else:
        print(
            f"偵測：YOLO（{args.model}），conf={args.conf}，"
            f"imgsz={args.imgsz}（每幀獨立，無 ID／追蹤）"
        )
    if args.track:
        print(
            "官方 track：同一支影片連續幀 model.track(persist=True)；"
            "靜態相機 gmc=none；BoT-SORT 不開短 ReID（長期 ID 交給 OSNet 圖庫＋座位）。"
        )
        print("畫面與 YOLO 分執行緒，預覽不因推論卡住。")
    print(f"參考點模式：{args.ref}（紅=foot，紫=head_drop）。按 q 結束，s 存圖。")
    print(f"預覽寬度固定 max-width={args.max_width}（影片與格子視窗皆鎖定畫面像素大小）")
    if args.stride > 1:
        print(
            f"跳幀：相機每幀都畫（框會預測跟上）；"
            f"YOLO 與格子佔用每 {args.stride} 幀才更新"
            f"（再加 cell-hold={args.cell_hold}，格子比較連續、比較不閃）。"
        )
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
    floor_overlay = FloorOverlayCache()
    if args.show_floor_grid:
        print("定位對照：相機只畫地上五點 A/B/C/D/O（無線）；右側細格保留，並強調同一組五點與有人色格。")

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
            show_floor_grid=args.show_floor_grid,
            floor_overlay=floor_overlay,
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
            f"本機影片：以約 {video_fps:.1f} fps 為播放上限；"
            "推論慢時播放會變慢，但不丟追蹤幀。"
        )

    stabilizer = CellStabilizer(args.cell_hold)
    confirmed_cells: set[tuple[int, int]] = set()
    box_coaster = DetectionCoaster()

    reid_encoder = None
    if args.track and args.reid:
        try:
            from reid_encoder import PersonReIDEncoder, resolve_reid_model

            model_name = resolve_reid_model(args.reid_model)
            print(f"載入 Re-ID：{model_name} …")
            reid_encoder = PersonReIDEncoder(model_name)
            print(
                f"Re-ID 就緒：{reid_encoder.model_name} "
                f"[backend={reid_encoder.backend}]（Stable-ID 長期圖庫）。"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Re-ID 載入失敗，改回 HSV：{exc}")
            reid_encoder = None

    coast_frames = max(int(args.id_coast), int(args.stride))
    id_mapper = StableIdMapper(
        max_dist_cm=args.id_max_dist,
        max_gap_frames=args.id_max_gap,
        max_speed_cm_s=args.id_max_speed,
        appear_thresh=args.appear_thresh,
        fps=video_fps if is_file_video or use_latest else 20.0,
        single_person=args.single_person,
        min_hits=args.min_hits,
        encoder=reid_encoder,
        coast_frames=coast_frames,
        sticky_frames=args.id_sticky,
        max_prototypes=args.max_prototypes,
        gallery_dir=None if args.no_gallery_dump else args.gallery_dir,
        review_dir=(
            args.review_dir
            if args.review_dump and not args.no_review_dump
            else None
        ),
        review_every=args.review_every,
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
                f"ID 穩定層：{appear_mode}｜YOLO框人 → temp 比對過去 ID 照片 → "
                f"命中沿用／連續追蹤換裝則存新照／都沒中才發新 ID；"
                f"appear≥{args.appear_thresh:.2f}，{proto_s}；"
                f"新 ID 需連續 {args.min_hits} 次仍對不上圖庫才發號；"
                "場上人數沒增加時優先接回空缺座位、不發新號。"
            )
        print(
            f"進出畫面：室內漏檢沿用約 {max(coast_frames, int(id_mapper.fps * 1.2))} 幀；"
            "貼畫面邊緣的框立刻清除（人已離開），不把殘框留在場上。"
        )
        if not args.no_gallery_dump:
            print(
                f"外貌圖庫：{args.gallery_dir}（ID***/；temp/current.jpg=當前比對圖；"
                f"每次重跑清空，只供即時比對）"
            )
        if id_mapper.review is not None:
            print(
                f"審查資料庫：{id_mapper.review.session_dir} "
                f"（每個 ID 約每 {args.review_every} 幀一張，背景 Pillow 寫檔，"
                "不參與即時比對；錯圖可之後刪）"
            )
        else:
            print("審查資料庫：關閉（預設；要存錯圖審查請加 --review-dump）")
        print(
            f"誤檢過濾：conf≥{args.conf}，min_bottom={args.min_bottom_ratio}，"
            f"min_aspect={args.min_aspect}，min_h={args.min_h_ratio}"
        )
        if args.log_id:
            print("ID 變化會印 [ID-CHANGE]（時刻 / elapsed / video / frame）。")
        print("格子視窗會標示各 ID 所在格（不同人不同底色）。")

    # Local files must be reproducible: process the fixed stride frames
    # synchronously so BoT-SORT always sees 1, 1+stride, 1+2*stride, ...
    # RTSP remains asynchronous/latest-frame because low live latency matters
    # more than replay determinism there.
    worker = (
        None
        if is_file_video
        else AsyncTrackWorker(model, h_mat, det_kw, id_mapper if args.track else None)
    )
    if is_file_video:
        print(
            f"本機影片：固定取樣第 1、{1 + args.stride}、"
            f"{1 + 2 * args.stride}…幀；同一影片重跑使用相同追蹤幀。"
        )
    save_path = Path(args.save_video).resolve() if args.save_video else None
    video_writer: cv2.VideoWriter | None = None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"錄製展示影片：{save_path}")
    try:
        frame_idx = 0
        last_dets: list[dict] = []
        last_timing: tuple[float, float] | None = None
        last_id_key: tuple[int, ...] | None = None
        last_obs_idx = -1
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
                ok, frame = cap.read()
                if ok:
                    frame_idx += 1
                if not ok or frame is None:
                    print("讀取結束或失敗。")
                    break

            run_detect = frame_idx == 1 or (frame_idx - 1) % args.stride == 0
            if is_file_video:
                if run_detect:
                    dets, detect_ms, locate_ms = detect_and_locate(
                        frame, model, h_mat, **det_kw
                    )
                    if det_kw.get("track"):
                        dets = id_mapper.apply(dets, frame_idx, frame=frame)
                    timing = (detect_ms, locate_ms)
                    det_idx = frame_idx
                else:
                    dets = last_dets
                    timing = last_timing
                    det_idx = last_obs_idx
            else:
                if run_detect:
                    worker.try_submit(frame, frame_idx)
                dets, timing, det_idx = worker.snapshot()
            if det_idx != last_obs_idx:
                last_obs_idx = det_idx
                last_dets = dets
                last_timing = None if args.no_timing else timing
                box_coaster.observe(last_dets, det_idx if det_idx else frame_idx)
                raw_cells = {d["cell"] for d in last_dets if d.get("cell") is not None}
                confirmed_cells = stabilizer.update(raw_cells)
                if det_kw.get("track") and args.log_id:
                    id_key = tuple(
                        sorted(
                            d["track_id"]
                            for d in last_dets
                            if d.get("track_id") is not None
                        )
                    )
                    if id_key != last_id_key:
                        log_id_change(det_idx, log_fps, t_play0, last_id_key, id_key)
                        last_id_key = id_key
            # Full-resolution resize + annotation + two GUI windows are
            # expensive on this 2880x1620 CPU-only setup. Keep every tracking
            # frame, but render a deterministic ~9 fps preview between them.
            # This changes display smoothness only, never the BoT-SORT input.
            render_frame = (
                save_path is not None
                or not is_file_video
                or run_detect
                or frame_idx == 1
                or (frame_idx - 1) % 3 == 0
            )
            if not render_frame:
                continue
            draw_dets = (
                box_coaster.extrapolate(last_dets, frame_idx) if last_dets else last_dets
            )
            vis, grid, logs = render_detection_view(
                frame,
                draw_dets,
                h_mat,
                args.valid_xmin,
                timing=last_timing,
                cached=det_idx != frame_idx,
                grid_cells=confirmed_cells,
                max_width=args.max_width,
                grid_cache=grid_cache,
                out_margin=args.out_margin,
                show_floor_grid=args.show_floor_grid,
                floor_overlay=floor_overlay,
            )
            if det_idx == frame_idx and not args.quiet:
                for line in logs:
                    print(line)
            if save_path is not None:
                demo_frame = stack_demo_views(vis, grid)
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(save_path),
                        fourcc,
                        video_fps,
                        (demo_frame.shape[1], demo_frame.shape[0]),
                    )
                    if not video_writer.isOpened():
                        raise RuntimeError(f"無法建立影片：{save_path}")
                video_writer.write(demo_frame)
            if not args.no_show:
                if not show_fixed_window(cam_win, vis) or not show_grid_window(
                    grid_win, grid
                ):
                    break

            if args.no_show:
                key = -1
            elif use_realtime:
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
        if video_writer is not None:
            video_writer.release()
        if worker is not None:
            worker.stop()
        if reader is not None:
            reader.release()
        else:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

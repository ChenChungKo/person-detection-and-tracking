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
DEFAULT_TRACKER = Path(__file__).resolve().parent / "trackers" / "bytetrack_stable.yaml"


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


class StableIdMapper:
    """Remap volatile tracker IDs using floor motion limits + appearance.

    Does NOT assume \"there is only one person\". A lost ID is reused only when
    a new detection is both:
      1) reachable in floor space given elapsed time and max walking speed, and
      2) similar in appearance (YOLO26 Re-ID embedding by default; HSV fallback).

    New tracks must appear for ``min_hits`` consecutive mapper updates before
    they receive a stable ID (and are shown), so one-frame FPs do not steal IDs.
    Stable IDs are never recycled in-session: a leaver may return and rematch
    to the same ID via floor distance + appearance.
    """

    def __init__(
        self,
        max_dist_cm: float = 320.0,
        max_gap_frames: int = 250,
        max_speed_cm_s: float = 180.0,
        appear_thresh: float = 0.50,
        fps: float = 20.0,
        single_person: bool = False,
        min_hits: int = 3,
        encoder=None,
    ) -> None:
        self.max_dist_cm = max_dist_cm
        self.max_gap_frames = max_gap_frames
        self.max_speed_cm_s = max_speed_cm_s
        self.appear_thresh = appear_thresh
        self.fps = max(1.0, fps)
        self.single_person = single_person  # optional demo-only; default off
        self.min_hits = max(1, int(min_hits))
        self.encoder = encoder
        self._raw_to_stable: dict[int, int] = {}
        self._stable: dict[int, dict] = {}  # sid -> frame, wx, wy, feat
        self._raw_hits: dict[int, int] = {}
        self._raw_last_frame: dict[int, int] = {}
        self._next_id = 1

    def _expire(self, frame_idx: int) -> None:
        # Keep stable ID records for the whole session so a returning person
        # can rematch; do not recycle numbers. Only drop stale *pending* hits
        # and break raw→stable links for tracker IDs not seen recently.
        stale_raw = [
            raw
            for raw, last in self._raw_last_frame.items()
            if frame_idx - int(last) > self.max_gap_frames
        ]
        for raw in stale_raw:
            self._raw_hits.pop(raw, None)
            self._raw_last_frame.pop(raw, None)
            self._raw_to_stable.pop(raw, None)

    def _alloc_stable_id(self) -> int:
        sid = self._next_id
        self._next_id += 1
        return sid

    @staticmethod
    def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)

    @staticmethod
    def appearance_feat(
        frame: np.ndarray,
        xyxy: tuple[int, int, int, int],
        encoder=None,
    ) -> np.ndarray | None:
        """Appearance vector: Re-ID embedding if encoder given, else HSV torso hist."""
        if encoder is not None:
            return encoder.embed_xyxy(frame, xyxy)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = xyxy
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
        if x2 - x1 < 8 or y2 - y1 < 16:
            return None
        # Focus on torso (skip head/feet) — more stable for clothing cue.
        y_a = y1 + int(0.20 * (y2 - y1))
        y_b = y1 + int(0.75 * (y2 - y1))
        x_a = x1 + int(0.15 * (x2 - x1))
        x_b = x1 + int(0.85 * (x2 - x1))
        crop = frame[y_a:y_b, x_a:x_b]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        n = float(np.linalg.norm(hist))
        if n < 1e-6:
            return None
        return hist / n

    @staticmethod
    def _appear_sim(a: np.ndarray | None, b: np.ndarray | None) -> float:
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b))

    def _reach_limit_cm(self, gap_frames: int) -> float:
        gap_s = max(0, gap_frames) / self.fps
        # reachable distance + small localization slack
        return min(self.max_dist_cm, self.max_speed_cm_s * gap_s + 60.0)

    def apply(
        self,
        dets: list[dict],
        frame_idx: int,
        frame: np.ndarray | None = None,
    ) -> list[dict]:
        self._expire(frame_idx)
        if not dets:
            return dets

        work = list(dets)
        # Demo-only shortcut (explicit --single-person). Off by default.
        if self.single_person and len(work) > 1:
            if self._stable:
                sid, meta = max(self._stable.items(), key=lambda kv: kv[1]["frame"])
                anchor = (float(meta["wx"]), float(meta["wy"]))
                work = [min(work, key=lambda d: self._dist(d.get("world", (1e9, 1e9)), anchor))]
            else:
                work = [max(work, key=lambda d: float(d.get("conf", 0.0)))]

        feats: list[np.ndarray | None] = []
        for det in work:
            if frame is None:
                feats.append(None)
            else:
                feats.append(
                    self.appearance_feat(frame, det["xyxy"], encoder=self.encoder)
                )

        assigned: dict[int, int] = {}
        used_sids: set[int] = set()

        # 1) Keep raw tracker ID links while that raw ID is still alive.
        for i, det in enumerate(work):
            raw = det.get("track_id")
            if raw is None:
                continue
            sid = self._raw_to_stable.get(int(raw))
            if sid is not None and sid in self._stable and sid not in used_sids:
                assigned[i] = sid
                used_sids.add(sid)

        # 2) Rematch by motion reachability + appearance (no one-person prior).
        unmatched_i = [i for i in range(len(work)) if i not in assigned]
        candidates = [sid for sid in self._stable if sid not in used_sids]
        pairs: list[tuple[float, int, int]] = []
        for i in unmatched_i:
            world = work[i].get("world")
            if world is None:
                continue
            for sid in candidates:
                meta = self._stable[sid]
                gap = frame_idx - int(meta["frame"])
                dist = self._dist(world, (float(meta["wx"]), float(meta["wy"])))
                limit = self._reach_limit_cm(gap)
                if self.single_person and len(work) == 1 and len(candidates) == 1:
                    limit = max(limit, self.max_dist_cm)
                if dist > limit:
                    continue
                sim = self._appear_sim(feats[i], meta.get("feat"))
                # Short gap + close on floor: motion alone is enough.
                short_close = gap <= 15 and dist <= 100.0
                if not short_close and sim < self.appear_thresh:
                    continue
                # Lower cost is better: prefer closer + more similar.
                cost = dist / max(limit, 1.0) + (1.0 - sim)
                pairs.append((cost, i, sid))
        pairs.sort()
        for _cost, i, sid in pairs:
            if i in assigned or sid in used_sids:
                continue
            assigned[i] = sid
            used_sids.add(sid)

        # 3) Confirm new tracks before issuing a recycled stable ID.
        out: list[dict] = []
        seen_raw: set[int] = set()
        for i, det in enumerate(work):
            d = dict(det)
            raw = d.get("track_id")
            if raw is not None:
                raw = int(raw)
                seen_raw.add(raw)
                self._raw_last_frame[raw] = frame_idx

            if i in assigned:
                sid = assigned[i]
            elif (
                self.single_person
                and len(work) == 1
                and len(self._stable) == 1
                and not used_sids
            ):
                sid = next(iter(self._stable))
            else:
                # Probation: need several consecutive hits before first ID.
                if raw is None:
                    # No tracker id (image / --no-track path): show immediately.
                    hits = self.min_hits
                else:
                    prev_last = self._raw_last_frame.get(raw)
                    # If this raw was missing last update, restart probation
                    # (handled by only incrementing when continuously seen via
                    # consecutive apply calls that include this raw).
                    hits = self._raw_hits.get(raw, 0) + 1
                    self._raw_hits[raw] = hits
                if hits < self.min_hits:
                    continue
                sid = self._alloc_stable_id()

            if raw is not None:
                self._raw_to_stable[raw] = sid
                self._raw_hits[raw] = max(self._raw_hits.get(raw, 0), self.min_hits)
            wx, wy = d.get("world", (0.0, 0.0))
            prev = self._stable.get(sid, {})
            feat = feats[i]
            old = prev.get("feat")
            if feat is not None and old is not None:
                feat = 0.8 * old + 0.2 * feat
                n = float(np.linalg.norm(feat))
                if n > 1e-6:
                    feat = feat / n
            elif feat is None:
                feat = old
            self._stable[sid] = {
                "frame": frame_idx,
                "wx": float(wx),
                "wy": float(wy),
                "feat": feat,
            }
            d["raw_track_id"] = raw
            d["track_id"] = sid
            out.append(d)

        # Reset hit streak for raw IDs not in this update (broken continuity).
        for raw in list(self._raw_hits.keys()):
            if raw not in seen_raw and raw not in self._raw_to_stable:
                self._raw_hits[raw] = 0
        return out


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
        help="do not print per-detection lines (less console lag)",
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
        default=320.0,
        help="cap on floor rematch distance (cm)",
    )
    p.add_argument(
        "--id-max-gap",
        type=int,
        default=250,
        help="max frames to remember a stable ID after last sighting",
    )
    p.add_argument(
        "--id-max-speed",
        type=float,
        default=180.0,
        help="max walk speed (cm/s) used to limit rematch distance by time gap",
    )
    p.add_argument(
        "--appear-thresh",
        type=float,
        default=0.45,
        help="min appearance cosine similarity (0-1) to reuse an ID after tracker switch",
    )
    p.add_argument(
        "--min-hits",
        type=int,
        default=3,
        help="new person must appear this many detect runs before getting an ID "
        "(hides one-frame false positives; default 3)",
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
            show_fixed_window(cam_win, vis)
            show_grid_window(grid_win, grid)
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
    )
    if args.track:
        if args.single_person:
            print("ID 穩定層：單人強制接回（demo，非真實辨識）")
        else:
            appear_mode = "Re-ID" if reid_encoder is not None else "HSV"
            print(
                f"ID 穩定層：移動距離上限 + {appear_mode}"
                f"（appear≥{args.appear_thresh:.2f}，max_speed={args.id_max_speed:.0f}cm/s，"
                f"min_hits={args.min_hits}）"
            )
        print(
            f"誤檢過濾：conf≥{args.conf}，min_bottom={args.min_bottom_ratio}，"
            f"min_aspect={args.min_aspect}，min_h={args.min_h_ratio}"
        )

    try:
        frame_idx = 0
        last_dets: list[dict] = []
        last_timing: tuple[float, float] | None = None
        last_id_key: tuple[int, ...] | None = None
        t_play0 = time.perf_counter()
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
            if run_detect and not args.quiet:
                id_key = tuple(
                    sorted(d["track_id"] for d in last_dets if d.get("track_id") is not None)
                )
                if id_key != last_id_key or logs:
                    if id_key != last_id_key:
                        print(f"[frame {frame_idx}] active IDs: {list(id_key) if id_key else '—'}")
                        last_id_key = id_key
                    for line in logs:
                        print(line)
            show_fixed_window(cam_win, vis)
            show_grid_window(grid_win, grid)

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

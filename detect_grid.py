"""Detect person (YOLO) + ground ref + light grid cell.

Ground reference:
  - foot / head_drop / auto / pose (see --ref)

Tracking (Ultralytics docs: https://docs.ultralytics.com/modes/track):
  - default: YOLO.track persist=True + BoT-SORT (trackers/botsort.yaml)
  - long-term IDs: Stable-ID + OSNet gallery (not the tracker yaml)
  - disable with --no-track

Usage:
  python detect_grid.py --source test/test.mp4 --ref auto
  python detect_grid.py --source test/test4.mp4 --ref pose
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
from PIL import Image, ImageDraw, ImageFont
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
# COCO pose: 11/12=hip, 13/14=knee, 15/16=ankle
_HIP_L, _HIP_R = 11, 12
_KNEE_L, _KNEE_R = 13, 14
_ANKLE_L, _ANKLE_R = 15, 16
REF_VIS = {
    "seat": {"color": (0, 200, 80), "tag": "座位"},
    "pose": {"color": (255, 255, 0), "tag": "腳踝"},
    "stand_drop": {"color": (0, 140, 255), "tag": "站立補腳"},
    "foot": {"color": (0, 0, 255), "tag": "框底"},
    "head_drop": {"color": (255, 0, 255), "tag": "推估"},
}
# COCO-17 skeleton (0-indexed), same topology as YOLO pose
_COCO_SKELETON: list[tuple[int, int]] = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
]
# BGR limb colors: legs, torso, arms, face (YOLO pose style)
_LIMB_COLORS: list[tuple[int, int, int]] = [
    (0, 165, 255),
    (0, 165, 255),
    (0, 165, 255),
    (0, 165, 255),
    (0, 165, 255),
    (255, 0, 255),
    (255, 0, 255),
    (255, 0, 255),
    (255, 128, 0),
    (255, 128, 0),
    (255, 128, 0),
    (255, 128, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
]
_KPT_COLORS: list[tuple[int, int, int]] = [
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (255, 200, 0),
    (255, 200, 0),
    (255, 128, 0),
    (255, 128, 0),
    (255, 128, 0),
    (255, 128, 0),
    (255, 0, 255),
    (255, 0, 255),
    (0, 165, 255),
    (0, 165, 255),
    (0, 165, 255),
    (0, 165, 255),
]


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
    """Homography is static: project the four floor marks once per preview size."""

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


def pose_model_name(detect_name: str) -> str:
    """Map yolo26s.pt → yolo26s-pose.pt (leave *-pose.pt unchanged)."""
    path = Path(detect_name)
    if "-pose" in path.name.lower():
        return detect_name
    return str(path.with_name(f"{path.stem}-pose{path.suffix or '.pt'}"))


def _kpt_visible(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray | None,
    idx: int,
    min_conf: float,
) -> tuple[float, float] | None:
    if kpts_xy is None or idx >= len(kpts_xy):
        return None
    x, y = float(kpts_xy[idx][0]), float(kpts_xy[idx][1])
    if not np.isfinite(x) or not np.isfinite(y) or (x <= 1.0 and y <= 1.0):
        return None
    conf = 1.0 if kpts_conf is None else float(kpts_conf[idx])
    if conf < min_conf:
        return None
    return x, y


def _pair_mid(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray | None,
    ia: int,
    ib: int,
    min_conf: float,
) -> tuple[float, float] | None:
    a = _kpt_visible(kpts_xy, kpts_conf, ia, min_conf)
    b = _kpt_visible(kpts_xy, kpts_conf, ib, min_conf)
    if a is not None and b is not None:
        return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
    return a if a is not None else b


def _shoulder_width(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray | None,
    min_conf: float,
    bbox_w: float,
) -> float:
    l_sh = _kpt_visible(kpts_xy, kpts_conf, 5, min_conf)
    r_sh = _kpt_visible(kpts_xy, kpts_conf, 6, min_conf)
    if l_sh is not None and r_sh is not None:
        sh_w = float(np.hypot(l_sh[0] - r_sh[0], l_sh[1] - r_sh[1]))
        if sh_w >= 0.25 * bbox_w:
            return sh_w
    return bbox_w


def ankle_ground_point(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray | None,
    min_conf: float,
) -> tuple[float, float] | None:
    """Midpoint of confident ankles, or the single visible ankle."""
    if kpts_xy is None or len(kpts_xy) <= _ANKLE_R:
        return None
    pts: list[tuple[float, float]] = []
    for idx in (_ANKLE_L, _ANKLE_R):
        x, y = float(kpts_xy[idx][0]), float(kpts_xy[idx][1])
        if not np.isfinite(x) or not np.isfinite(y) or (x <= 1.0 and y <= 1.0):
            continue
        conf = 1.0 if kpts_conf is None else float(kpts_conf[idx])
        if conf >= min_conf:
            pts.append((x, y))
    if not pts:
        return None
    if len(pts) == 1:
        return pts[0]
    return (0.5 * (pts[0][0] + pts[1][0]), 0.5 * (pts[0][1] + pts[1][1]))


def box_iou(
    ax1: float, ay1: float, ax2: float, ay2: float,
    bx1: float, by1: float, bx2: float, by2: float,
) -> float:
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


def match_pose_keypoints(
    det_xyxy: np.ndarray,
    pose_result,
    min_iou: float = 0.25,
) -> list[tuple[np.ndarray, np.ndarray | None] | None]:
    """Map each detect/track box index → (kpts_xy, kpts_conf) from pose boxes."""
    n = len(det_xyxy)
    out: list[tuple[np.ndarray, np.ndarray | None] | None] = [None] * n
    if pose_result is None or pose_result.boxes is None or len(pose_result.boxes) == 0:
        return out
    pk = getattr(pose_result, "keypoints", None)
    if pk is None or pk.xy is None:
        return out
    pboxes = pose_result.boxes
    for i in range(n):
        x1, y1, x2, y2 = det_xyxy[i].tolist()
        best_j = -1
        best_iou = min_iou
        for j in range(len(pboxes)):
            if int(pboxes.cls[j].item()) != 0:
                continue
            px1, py1, px2, py2 = pboxes.xyxy[j].tolist()
            iou = box_iou(x1, y1, x2, y2, px1, py1, px2, py2)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j < 0:
            continue
        xy = pk.xy[best_j].detach().cpu().numpy()
        confs = None
        if pk.conf is not None:
            confs = pk.conf[best_j].detach().cpu().numpy()
        out[i] = (xy, confs)
    return out


def classify_sitting(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray | None,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    min_conf: float,
) -> bool:
    """Recognize sitting from bent thighs, not from bbox height or visible ankles."""
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    aspect = bh / bw
    hips = _pair_mid(kpts_xy, kpts_conf, _HIP_L, _HIP_R, min_conf)
    knees = _pair_mid(kpts_xy, kpts_conf, _KNEE_L, _KNEE_R, min_conf)
    shoulders = _pair_mid(kpts_xy, kpts_conf, 5, 6, min_conf)

    if hips is None or knees is None:
        return False

    thigh_dy = knees[1] - hips[1]
    thigh_dx = abs(knees[0] - hips[0])
    pair_horizontal = (
        thigh_dx >= 1.15 * max(abs(thigh_dy), 1.0)
        and thigh_dx >= 0.07 * bh
    )

    # Pair midpoints can cancel when legs point in opposite directions. Accept
    # either left/right thigh when it is clearly more horizontal than vertical.
    side_horizontal = False
    for hip_idx, knee_idx in ((_HIP_L, _KNEE_L), (_HIP_R, _KNEE_R)):
        hip = _kpt_visible(kpts_xy, kpts_conf, hip_idx, min_conf)
        knee = _kpt_visible(kpts_xy, kpts_conf, knee_idx, min_conf)
        if hip is None or knee is None:
            continue
        dx = abs(knee[0] - hip[0])
        dy = abs(knee[1] - hip[1])
        if dx >= 1.05 * max(dy, 1.0) and dx >= 0.07 * bh:
            side_horizontal = True
            break

    if (pair_horizontal or side_horizontal) and aspect < 2.7:
        return True

    # A folded thigh can be nearly zero-length after left/right averaging.
    # Keep this weaker cue only for compact boxes with a normal visible torso.
    if shoulders is not None:
        torso_dy = hips[1] - shoulders[1]
        folded = (
            torso_dy > 8.0
            and abs(thigh_dy) < 0.45 * torso_dy
            and thigh_dx >= 0.04 * bh
        )
        if folded and aspect < 1.9:
            return True
    return False


def standing_drop_point(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray | None,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_h: int,
    aspect: float,
    min_conf: float,
    drop_ratio: float,
) -> tuple[tuple[float, float], list[tuple[float, float]]] | None:
    """Estimate floor under a standing person whose legs are hidden (table / crop)."""
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    y_max = float(frame_h - 1)
    shoulders = _pair_mid(kpts_xy, kpts_conf, 5, 6, min_conf)
    hips = _pair_mid(kpts_xy, kpts_conf, _HIP_L, _HIP_R, min_conf)
    nose = _kpt_visible(kpts_xy, kpts_conf, 0, min_conf)
    head_y = nose[1] if nose is not None else float(y1)
    sh_w = _shoulder_width(kpts_xy, kpts_conf, min_conf, bw)

    if hips is not None and shoulders is not None:
        torso = hips[1] - shoulders[1]
        src = [hips]
        if torso > 10.0:
            foot_x, foot_y = hips[0], hips[1] + torso * drop_ratio
        else:
            foot_x, foot_y = hips[0], head_y + sh_w * aspect
    elif shoulders is not None:
        src = [shoulders]
        foot_x, foot_y = shoulders[0], head_y + sh_w * aspect
    else:
        return None

    foot_y = max(foot_y, float(y2))
    cap = min(y_max, float(y2) + 1.80 * bh)
    foot_y = min(foot_y, cap)
    foot_x = min(max(foot_x, float(x1)), float(x2))
    return (foot_x, foot_y), src


def lower_body_hidden(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray | None,
    y2: float,
    min_conf: float,
) -> bool | None:
    """True if visible length below hip is too short for standing legs.

    None if hips/shoulders missing. Full-body standing: below-hip ≈ 1.5–1.8× torso.
    Table cut: hip already near box bottom.
    """
    hips = _pair_mid(kpts_xy, kpts_conf, _HIP_L, _HIP_R, min_conf)
    shoulders = _pair_mid(kpts_xy, kpts_conf, 5, 6, min_conf)
    if hips is None or shoulders is None:
        return None
    torso = hips[1] - shoulders[1]
    if torso < 10.0:
        return None
    # Standing legs ≈ 1.6× torso; table cut leaves ≲ one torso below the hip.
    return (y2 - hips[1]) < 1.20 * torso


def ankles_trusted(
    ankle: tuple[float, float],
    hips: tuple[float, float] | None,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_h: int,
    truncated: bool,
    legs_cut: bool | None,
) -> bool:
    """Reject pose ankles glued to a mid-frame box bottom (table-edge fakes)."""
    bh = max(1.0, y2 - y1)
    ay = ankle[1]
    if ay < y1 + 0.55 * bh:
        return False
    at_frame_bottom = y2 >= 0.90 * frame_h
    near_box_bottom = ay >= y2 - 0.12 * bh
    if legs_cut is True:
        return False
    if hips is not None and ay < hips[1] + 0.28 * bh and not at_frame_bottom:
        return False
    if truncated and near_box_bottom and not at_frame_bottom:
        return False
    return True


class PoseTrackHistory:
    """Complete occluded feet from reliable poses previously seen on each track.

    This follows the safer PETL4SD-style order: current visible joints, temporal
    body proportions, then bbox fallback. It never invents a standing extension
    for a track that has not first supplied a trustworthy full-body sample.
    """

    def __init__(self, sit_confirm: int = 3, max_history_age: int = 40) -> None:
        self.sit_confirm = max(1, int(sit_confirm))
        self.max_history_age = max(1, int(max_history_age))
        self._tracks: dict[int, dict] = {}

    @staticmethod
    def _ema(old, new, alpha: float = 0.25):
        if old is None:
            return new
        if isinstance(new, tuple):
            return (
                (1.0 - alpha) * old[0] + alpha * new[0],
                (1.0 - alpha) * old[1] + alpha * new[1],
            )
        return (1.0 - alpha) * old + alpha * new

    def resolve(
        self,
        track_id: int | None,
        kpts_xy: np.ndarray,
        kpts_conf: np.ndarray | None,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        frame_h: int,
        min_conf: float,
    ) -> tuple[tuple[float, float] | None, str, list[tuple[float, float]]]:
        if track_id is None:
            return None, "foot", []

        state = self._tracks.setdefault(
            int(track_id),
            {
                "sit_hits": 0,
                "full_hits": 0,
                "history_age": self.max_history_age + 1,
                "torso": None,
                "hip_to_foot": None,
                "shoulder_to_foot": None,
            },
        )
        state["history_age"] += 1

        hips = _pair_mid(kpts_xy, kpts_conf, _HIP_L, _HIP_R, min_conf)
        shoulders = _pair_mid(kpts_xy, kpts_conf, 5, 6, min_conf)
        ankle = ankle_ground_point(kpts_xy, kpts_conf, min_conf)
        bh = max(1.0, y2 - y1)
        sit_evidence = classify_sitting(
            kpts_xy, kpts_conf, x1, y1, x2, y2, min_conf
        )
        if sit_evidence:
            state["sit_hits"] = min(self.sit_confirm, state["sit_hits"] + 1)
        else:
            state["sit_hits"] = max(0, state["sit_hits"] - 1)

        legs_cut = lower_body_hidden(kpts_xy, kpts_conf, y2, min_conf)
        torso = None
        if hips is not None and shoulders is not None:
            torso = hips[1] - shoulders[1]

        # A training sample must show a plausible full leg inside the person
        # box. This excludes YOLO pose ankles hallucinated on a table edge.
        full_body = (
            not sit_evidence
            and legs_cut is False
            and torso is not None
            and torso > 10.0
            and hips is not None
            and shoulders is not None
            and ankle is not None
            and ankle[1] <= y2 + 0.05 * bh
            and ankle[1] - hips[1] >= 1.10 * torso
        )
        if full_body:
            state["full_hits"] += 1
            state["history_age"] = 0
            state["torso"] = self._ema(state["torso"], torso)
            state["hip_to_foot"] = self._ema(
                state["hip_to_foot"],
                (ankle[0] - hips[0], ankle[1] - hips[1]),
            )
            state["shoulder_to_foot"] = self._ema(
                state["shoulder_to_foot"],
                (ankle[0] - shoulders[0], ankle[1] - shoulders[1]),
            )

        # Bent legs immediately block a standing extension. Wait for a few
        # consistent observations before showing the green seated reference.
        if sit_evidence:
            if state["sit_hits"] >= self.sit_confirm and hips is not None:
                hx = min(max(hips[0], x1), x2)
                return (hx, float(y2)), "seat", [hips]
            return None, "foot", []

        if full_body and ankle is not None:
            ax = min(max(ankle[0], x1), x2)
            ay = min(max(ankle[1], y1), float(frame_h - 1))
            return (ax, ay), "pose", []

        usable_history = (
            state["full_hits"] >= 2
            and state["history_age"] <= self.max_history_age
            and state["torso"] is not None
        )
        if legs_cut is True and usable_history:
            scale = 1.0
            if torso is not None and torso > 10.0:
                scale = float(np.clip(torso / state["torso"], 0.75, 1.35))
            src = None
            foot = None
            if hips is not None and state["hip_to_foot"] is not None:
                dx, dy = state["hip_to_foot"]
                src = hips
                foot = (hips[0] + dx * scale, hips[1] + dy * scale)
            elif shoulders is not None and state["shoulder_to_foot"] is not None:
                dx, dy = state["shoulder_to_foot"]
                src = shoulders
                foot = (shoulders[0] + dx * scale, shoulders[1] + dy * scale)
            if foot is not None and src is not None:
                fx = min(max(foot[0], x1), x2)
                fy = min(max(foot[1], y2), float(frame_h - 1))
                return (fx, fy), "stand_drop", [src]

        # Trust a current ankle only when it resembles a complete visible leg.
        if ankle is not None and legs_cut is False:
            ax = min(max(ankle[0], x1), x2)
            ay = min(max(ankle[1], y1), float(frame_h - 1))
            return (ax, ay), "pose", []

        # No trustworthy history: explicitly decline to extrapolate.
        return None, "foot", []


def pose_ground_from_keypoints(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray | None,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_h: int,
    kpt_conf: float,
    aspect: float,
    drop_ratio: float,
) -> tuple[tuple[float, float] | None, str, list[tuple[float, float]]]:
    """Sit: hip X + box bottom. Stand + hidden legs: drop to floor. Else ankles / box."""
    hips = _pair_mid(kpts_xy, kpts_conf, _HIP_L, _HIP_R, kpt_conf)
    sitting = classify_sitting(kpts_xy, kpts_conf, x1, y1, x2, y2, kpt_conf)
    if sitting:
        if hips is not None:
            hx = min(max(hips[0], x1), x2)
            return (hx, float(y2)), "seat", [hips]
        return None, "foot", []

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    sh_w = _shoulder_width(kpts_xy, kpts_conf, kpt_conf, bw)
    truncated = (sh_w * aspect) > 1.25 * bh
    mid_frame = y2 < 0.90 * frame_h
    legs_cut = lower_body_hidden(kpts_xy, kpts_conf, y2, kpt_conf)

    raw = ankle_ground_point(kpts_xy, kpts_conf, kpt_conf)
    trusted = raw is not None and ankles_trusted(
        raw, hips, x1, y1, x2, y2, frame_h, truncated, legs_cut
    )
    if trusted and raw is not None:
        ax = min(max(raw[0], x1), x2)
        ay = min(max(raw[1], float(y1)), float(y2) + 8.0)
        return (ax, min(ay, float(frame_h - 1))), "pose", []

    # Table / crop: hip too close to box bottom, or box shorter than standing height.
    can_drop = legs_cut is True or (truncated and ((bh / bw) >= 1.25 or mid_frame))
    if not can_drop and mid_frame and raw is not None and not trusted:
        can_drop = True
    if can_drop:
        dropped = standing_drop_point(
            kpts_xy,
            kpts_conf,
            x1,
            y1,
            x2,
            y2,
            frame_h,
            aspect,
            kpt_conf,
            drop_ratio,
        )
        if dropped is not None:
            return dropped[0], "stand_drop", dropped[1]

    if hips is not None:
        hx = min(max(hips[0], x1), x2)
        return (hx, float(y2)), "foot", []
    return None, "foot", []


def draw_pose_skeleton(
    vis: np.ndarray,
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray | None,
    min_conf: float = 0.25,
    line_scale: float = 1.0,
) -> None:
    """Draw COCO-17 skeleton on the preview frame (YOLO pose style)."""
    thick = max(2, int(round(3 * line_scale)))
    radius = max(3, int(round(4 * line_scale)))
    for edge_i, (a, b) in enumerate(_COCO_SKELETON):
        pa = _kpt_visible(kpts_xy, kpts_conf, a, min_conf)
        pb = _kpt_visible(kpts_xy, kpts_conf, b, min_conf)
        if pa is None or pb is None:
            continue
        color = _LIMB_COLORS[edge_i] if edge_i < len(_LIMB_COLORS) else (200, 200, 200)
        cv2.line(
            vis,
            (int(round(pa[0])), int(round(pa[1]))),
            (int(round(pb[0])), int(round(pb[1]))),
            color,
            thick,
            cv2.LINE_AA,
        )
    for idx in range(min(len(kpts_xy), len(_KPT_COLORS))):
        pt = _kpt_visible(kpts_xy, kpts_conf, idx, min_conf)
        if pt is None:
            continue
        cv2.circle(
            vis,
            (int(round(pt[0])), int(round(pt[1]))),
            radius,
            _KPT_COLORS[idx],
            -1,
            cv2.LINE_AA,
        )


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
    kpt_conf: float = 0.35,
    stand_drop_ratio: float = 1.7,
    pose_result=None,
    pose_history: PoseTrackHistory | None = None,
) -> list[dict]:
    """Return person ground-ref points from YOLO detect/track boxes."""
    out: list[dict] = []
    if result.boxes is None or len(result.boxes) == 0:
        return out
    boxes = result.boxes
    has_ids = boxes.id is not None
    kpts = getattr(result, "keypoints", None)
    pose_kpts = (
        match_pose_keypoints(boxes.xyxy, pose_result)
        if mode == "pose" and pose_result is not None
        else None
    )
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
        pose_pt = None
        used = "foot"
        ref_src: list[tuple[float, float]] = []
        kpts_xy_store = None
        kpts_conf_store = None
        if mode == "pose":
            if pose_kpts is not None and pose_kpts[i] is not None:
                xy, confs = pose_kpts[i]
                kpts_xy_store = xy
                kpts_conf_store = confs
            elif kpts is not None and kpts.xy is not None and i < len(kpts.xy):
                xy = kpts.xy[i].detach().cpu().numpy()
                confs = None
                if kpts.conf is not None:
                    confs = kpts.conf[i].detach().cpu().numpy()
                kpts_xy_store = xy
                kpts_conf_store = confs
            if kpts_xy_store is not None:
                if pose_history is not None:
                    got, used, ref_src = pose_history.resolve(
                        track_id,
                        kpts_xy_store,
                        kpts_conf_store,
                        x1,
                        y1,
                        x2,
                        y2,
                        frame_h,
                        kpt_conf,
                    )
                else:
                    got, used, ref_src = pose_ground_from_keypoints(
                        kpts_xy_store,
                        kpts_conf_store,
                        x1,
                        y1,
                        x2,
                        y2,
                        frame_h,
                        kpt_conf,
                        aspect,
                        stand_drop_ratio,
                    )
                if got is not None:
                    pose_pt = got
        if pose_pt is not None:
            ref_x, ref_y = pose_pt[0], pose_pt[1]
        else:
            ref_x, ref_y, used = estimate_ref_point(
                x1, y1, x2, y2, frame_h, "auto" if mode == "pose" else mode, aspect, truncate_ratio
            )
        item = {
            "xyxy": (int(x1), int(y1), int(x2), int(y2)),
            "foot": (ref_x, ref_y),
            "head": (0.5 * (x1 + x2), float(y1)),
            "mode": used,
            "conf": conf,
            "track_id": track_id,
        }
        if ref_src:
            item["ref_src"] = ref_src
        if kpts_xy_store is not None:
            item["kpts_xy"] = kpts_xy_store
            if kpts_conf_store is not None:
                item["kpts_conf"] = kpts_conf_store
        out.append(item)
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
        if "kpts_xy" in det:
            k = np.asarray(det["kpts_xy"], dtype=np.float64)
            k = k.copy()
            k[:, 0] *= scale
            k[:, 1] *= scale
            d["kpts_xy"] = k
        if "ref_src" in det:
            d["ref_src"] = [(kx * scale, ky * scale) for kx, ky in det["ref_src"]]
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
            if "kpts_xy" in d:
                k = np.asarray(d["kpts_xy"], dtype=np.float64)
                k = k.copy()
                k[:, 0] += vcx * steps
                k[:, 1] += vcy * steps
                d["kpts_xy"] = k
            if "ref_src" in d:
                d["ref_src"] = [
                    (kx + vcx * steps, ky + vcy * steps) for kx, ky in d["ref_src"]
                ]
            out.append(d)
        return out


def draw_ref_guide(
    vis: np.ndarray,
    fx: float,
    fy: float,
    used: str,
    ref_src: list[tuple[float, float]] | None,
    fs: float,
) -> None:
    """Color-coded foot dot; dashed guide from hip/shoulder when estimated."""
    style = REF_VIS.get(used, REF_VIS["foot"])
    color = style["color"]
    r = max(6, int(5 * fs))
    cv2.circle(vis, (int(fx), int(fy)), r, color, -1, cv2.LINE_AA)
    cv2.circle(vis, (int(fx), int(fy)), r + 2, (255, 255, 255), 1, cv2.LINE_AA)
    if used in ("seat", "stand_drop") and ref_src:
        for kx, ky in ref_src:
            _draw_dashed_line(
                vis,
                (int(kx), int(ky)),
                (int(fx), int(fy)),
                color,
                max(2, int(round(2 * fs))),
            )


def draw_ref_legend(vis: np.ndarray, fs: float) -> None:
    """One-line color key at the top of the camera view."""
    keys = ("seat", "pose", "stand_drop", "foot", "head_drop")
    font_px = max(16, int(18 * fs))
    pad_x = max(10, int(10 * fs))
    pad_y = max(6, int(6 * fs))
    gap = max(16, int(16 * fs))
    r = max(6, int(5 * fs))
    font = _legend_font(font_px)
    labels = [REF_VIS[k]["tag"] for k in keys]
    widths: list[int] = []
    for label in labels:
        bbox = font.getbbox(label)
        widths.append(max(1, bbox[2] - bbox[0]))
    bar_w = pad_x
    for tw in widths:
        bar_w += r * 2 + 8 + tw + gap
    bar_w = bar_w - gap + pad_x
    bar_h = r * 2 + pad_y * 2
    bar_w = min(bar_w, vis.shape[1] - 16)
    bar = Image.new("RGBA", (bar_w, bar_h), (0, 0, 0, 140))
    draw = ImageDraw.Draw(bar)
    x = pad_x
    cy = bar_h // 2
    for key, label, tw in zip(keys, labels, widths):
        color = REF_VIS[key]["color"]
        rgb = (color[2], color[1], color[0], 255)
        draw.ellipse((x, cy - r, x + r * 2, cy + r), fill=rgb, outline=(255, 255, 255, 220))
        draw.text((x + r * 2 + 6, cy - font_px // 2 - 1), label, font=font, fill=(255, 255, 255, 255))
        x += r * 2 + 8 + tw + gap
    rgb = np.asarray(bar.convert("RGB"))
    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    alpha = np.asarray(bar.split()[-1], dtype=np.float32) / 255.0
    y1, x1 = 8, 8
    y2, x2 = y1 + bar_h, x1 + bar_w
    roi = vis[y1:y2, x1:x2]
    if roi.shape[0] != bar_h or roi.shape[1] != bar_w:
        return
    a = alpha[..., None]
    vis[y1:y2, x1:x2] = (overlay.astype(np.float32) * a + roi.astype(np.float32) * (1.0 - a)).astype(
        np.uint8
    )


def _legend_font(size_px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\msyhbd.ttc"),
        Path(r"C:\Windows\Fonts\mingliu.ttc"),
    ):
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size_px)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_dashed_line(
    vis: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: tuple[int, int, int],
    thick: int,
    dash: int = 8,
    gap: int = 6,
) -> None:
    x0, y0 = p0
    x1, y1 = p1
    dist = float(np.hypot(x1 - x0, y1 - y0))
    if dist < 1.0:
        return
    dx = (x1 - x0) / dist
    dy = (y1 - y0) / dist
    step = dash + gap
    n = int(dist / step) + 1
    for i in range(n):
        t0 = i * step
        t1 = min(dist, t0 + dash)
        if t0 >= dist:
            break
        a = (int(x0 + dx * t0), int(y0 + dy * t0))
        b = (int(x0 + dx * t1), int(y0 + dy * t1))
        cv2.line(vis, a, b, color, thick, cv2.LINE_AA)


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
    show_pose_skeleton: bool = False,
    kpt_draw_conf: float = 0.25,
) -> tuple[np.ndarray, set[tuple[int, int]], list[str]]:
    vis = frame  # caller passes a writable preview-sized copy
    cells: set[tuple[int, int]] = set()
    logs: list[str] = []
    fs = max(0.75, frame.shape[1] / 1280.0)
    thick = max(2, int(round(2 * fs)))

    for det in detections:
        if show_pose_skeleton and "kpts_xy" in det:
            draw_pose_skeleton(
                vis,
                np.asarray(det["kpts_xy"]),
                np.asarray(det["kpts_conf"]) if det.get("kpts_conf") is not None else None,
                min_conf=kpt_draw_conf,
                line_scale=fs,
            )

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
        draw_ref_guide(
            vis,
            fx,
            fy,
            used,
            det.get("ref_src"),
            fs,
        )

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

    draw_ref_legend(vis, fs)
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
    kpt_conf: float = 0.35,
    stand_drop_ratio: float = 1.7,
    pose_model: YOLO | None = None,
    pose_history: PoseTrackHistory | None = None,
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
    pose_result = None
    if ref == "pose" and pose_model is not None:
        pose_result = pose_model.predict(frame, **infer_kw)[0]
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
        kpt_conf=kpt_conf,
        stand_drop_ratio=stand_drop_ratio,
        pose_result=pose_result,
        pose_history=pose_history,
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
        self._pose_model = det_kw.get("pose_model")
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
    show_pose_skeleton: bool = False,
    kpt_draw_conf: float = 0.25,
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
        show_cell_label=False,
        show_pose_skeleton=show_pose_skeleton,
        kpt_draw_conf=kpt_draw_conf,
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
    p.add_argument("--model", default="yolo26s.pt", help="Ultralytics weights (yolo26s.pt or yolo26s-pose.pt)")
    p.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="YOLO confidence (default 0.45; lower to 0.1 keeps more seated/far people, "
        "higher drops monitors and door-edge fragments)",
    )
    p.add_argument(
        "--ref",
        choices=["auto", "foot", "head_drop", "pose"],
        default="auto",
        help="auto: bbox bottom, head_drop if cut by frame bottom; "
        "pose: sit = hip X + box bottom; stand + hidden legs = drop to floor; "
        "visible ankles otherwise",
    )
    p.add_argument(
        "--kpt-conf",
        type=float,
        default=0.35,
        help="min keypoint confidence to use an ankle when --ref pose",
    )
    p.add_argument(
        "--show-pose-skeleton",
        action="store_true",
        default=None,
        help="draw COCO-17 skeleton on camera view (default on when --ref pose)",
    )
    p.add_argument(
        "--no-pose-skeleton",
        dest="show_pose_skeleton",
        action="store_false",
        help="hide pose skeleton overlay",
    )
    p.add_argument(
        "--kpt-draw-conf",
        type=float,
        default=0.25,
        help="min keypoint confidence to draw skeleton joints/lines",
    )
    p.add_argument(
        "--stand-drop-ratio",
        type=float,
        default=1.7,
        help="single-frame/no-track fallback only; tracked video uses learned per-ID proportions",
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
        help="overlay four floor marks (A, B, C + center O) on camera and bird-eye views",
    )
    p.add_argument(
        "--no-floor-grid",
        dest="show_floor_grid",
        action="store_false",
        help="hide the four floor marks on camera and bird-eye views",
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
    pose_model: YOLO | None = None
    pose_history: PoseTrackHistory | None = None
    if args.ref == "pose":
        pose_path = pose_model_name(args.model)
        if Path(pose_path).name.lower() == Path(args.model).name.lower():
            raise SystemExit(
                "--ref pose 需要 detect 權重（例如 yolo26s.pt）；"
                f"腳踝會另跑 {pose_path}，請勿把 --model 設成 pose 版。"
            )
        pose_model = YOLO(pose_path)
        if args.track:
            pose_history = PoseTrackHistory(sit_confirm=3, max_history_age=40)
        print(
            f"--ref pose：偵測／追蹤仍用 {args.model}，姿態另跑 {pose_path}"
            "（遮擋補腳只使用同一追蹤 ID 先前可靠的完整站姿）"
        )

    source = args.source
    is_image = Path(source).exists() and Path(source).suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    if args.show_pose_skeleton is None:
        args.show_pose_skeleton = args.ref == "pose"

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
    print(
        f"參考點模式：{args.ref}（畫面上方圖例：綠=座位，青=腳踝，橘=站立補腳，紅=框底，紫=推估）。按 q 結束，s 存圖。"
    )
    if args.ref == "pose":
        print(
            "pose：坐姿需連續 3 次證據；站立補腳需同一 raw track 先累積至少 2 次"
            "完整站姿，之後才用該人的歷史身體比例補腳；無歷史一律退回框底。"
        )
        if args.show_pose_skeleton:
            print(
                f"骨架：COCO-17 關節＋連線（kpt-draw-conf≥{args.kpt_draw_conf:g}）；"
                "--no-pose-skeleton 可關閉。"
            )
    print(f"預覽寬度固定 max-width={args.max_width}（視窗可拖曳縮放）")
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
        kpt_conf=args.kpt_conf,
        stand_drop_ratio=args.stand_drop_ratio,
        pose_model=pose_model,
        pose_history=pose_history,
    )
    grid_cache = GridCache()
    floor_overlay = FloorOverlayCache()
    if args.show_floor_grid:
        print("定位對照：四點 A/B/C/O（可手動選：python pick_floor_marks.py）；人框只標 ID。")

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
            show_pose_skeleton=args.show_pose_skeleton,
            kpt_draw_conf=args.kpt_draw_conf,
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
                "桌後漏檢接回空缺、不發新號；走出畫面後外貌明顯不像才發新號。"
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
                show_pose_skeleton=args.show_pose_skeleton,
                kpt_draw_conf=args.kpt_draw_conf,
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

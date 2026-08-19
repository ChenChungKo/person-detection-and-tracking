"""Stable person ID remapping — gallery-first Re-ID.

Intended pipeline (matches product design):
  1) YOLO boxes a full person
  2) Crop → temp (current look) + Re-ID embedding
  3) Compare temp against every past ID photo / prototype
  4) Match → reuse that ID
  5) Continuous track (bbox never broke) + clothes change → save a new photo
     on the SAME ID (do not mint)
  6) Only if temp matches NO past ID → mint a new ID + first.jpg
"""

from __future__ import annotations

import queue
import shutil
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class ReviewDumper:
    """Background file dump for later human cleanup. Not used for live Re-ID.

    The tracker thread only copies a numpy crop onto a drop-if-full queue.
    Resize, JPEG, and disk I/O run on a side thread with Pillow — never
    OpenCV — so YOLO/BoT-SORT is not slowed and does not share cv2 state.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        every_frames: int = 10,
        max_edge: int = 320,
        jpeg_quality: int = 85,
    ) -> None:
        self.every_frames = max(1, int(every_frames))
        self.max_edge = max(64, int(max_edge))
        self.jpeg_quality = int(jpeg_quality)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(root) / stamp
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._last_frame: dict[int, int] = {}
        self._q: queue.Queue[tuple[Path, np.ndarray] | None] = queue.Queue(maxsize=24)
        self._thread = threading.Thread(
            target=self._writer, name="review-dump", daemon=True
        )
        self._thread.start()

    def _writer(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            path, crop = item
            try:
                rgb = crop[:, :, ::-1]
                img = Image.fromarray(rgb)
                img.thumbnail(
                    (self.max_edge, self.max_edge), Image.Resampling.BILINEAR
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                img.save(
                    path,
                    format="JPEG",
                    quality=self.jpeg_quality,
                    optimize=False,
                )
            except Exception:
                continue

    def maybe_save(
        self,
        sid: int,
        frame: np.ndarray,
        xyxy: tuple[int, int, int, int],
        frame_idx: int,
    ) -> None:
        last = self._last_frame.get(sid, -10**9)
        if frame_idx - last < self.every_frames:
            return
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 - x1 < 8 or y2 - y1 < 16:
            return
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return
        # Cheap integer downsample so the queue stays small; Pillow does
        # the final resize. No OpenCV on this thread.
        edge = max(crop.shape[0], crop.shape[1])
        step = max(1, edge // (self.max_edge * 2))
        crop = np.ascontiguousarray(crop[::step, ::step])
        path = self.session_dir / f"ID{sid:03d}" / f"f{frame_idx:05d}.jpg"
        try:
            self._q.put_nowait((path, crop))
            self._last_frame[sid] = frame_idx
        except queue.Full:
            return


class StableIdMapper:
    """Remap ByteTrack IDs using a session appearance gallery.

    Leave duration is unrestricted within one run. Reappearing people are
    rebound by comparing the current ``temp`` look to past ID photos
    (embeddings). Each person keeps **multiple appearance prototypes**
    (e.g. with / without jacket). A clothing change on a **continuous**
    track records a new photo but keeps the same ID.

    A **new** ID is issued only after confirmed hits still fail every past
    gallery photo. Among near-equal gallery hits, the **most recently seen**
    ID wins.
    """

    def __init__(
        self,
        max_dist_cm: float = 400.0,
        max_gap_frames: int = 400,
        max_speed_cm_s: float = 200.0,
        appear_thresh: float = 0.32,
        fps: float = 20.0,
        single_person: bool = False,
        min_hits: int = 5,
        encoder=None,
        coast_frames: int = 8,
        sticky_frames: int = 45,
        soft_appear_thresh: float | None = None,
        max_prototypes: int = 0,
        proto_new_thresh: float | None = None,
        gallery_dir: str | Path | None = None,
        gallery_latest_every: int = 20,
        review_dir: str | Path | None = None,
        review_every: int = 10,
    ) -> None:
        self.max_dist_cm = max_dist_cm
        self.max_gap_frames = max_gap_frames
        self.max_speed_cm_s = max_speed_cm_s
        self.appear_thresh = appear_thresh
        # Long-leave rematch may use this slightly softer bar (short-leave still
        # uses appear_thresh + spatial gate so two desks do not merge).
        self.soft_appear_thresh = (
            appear_thresh * 0.78 if soft_appear_thresh is None else soft_appear_thresh
        )
        self.fps = max(1.0, fps)
        self.single_person = single_person
        self.min_hits = max(1, int(min_hits))
        self.encoder = encoder
        self.coast_frames = max(0, int(coast_frames))
        self.sticky_frames = max(0, int(sticky_frames))
        # 0 / negative = unlimited prototypes (new looks keep appending).
        self.max_prototypes = int(max_prototypes)
        # If current look is below this vs all stored looks → add a new prototype
        # (clothing change), instead of only EMA-washing the old jacket look away.
        self.proto_new_thresh = (
            appear_thresh + 0.08 if proto_new_thresh is None else proto_new_thresh
        )
        self.gallery_latest_every = max(1, int(gallery_latest_every))
        self.gallery_dir = Path(gallery_dir) if gallery_dir else None
        self.review = (
            ReviewDumper(review_dir, every_frames=review_every)
            if review_dir
            else None
        )
        self._raw_to_stable: dict[int, int] = {}
        self._stable: dict[int, dict] = {}
        self._raw_hits: dict[int, int] = {}
        self._raw_last_frame: dict[int, int] = {}
        self._raw_miss: dict[int, int] = {}
        self._pending_feat: dict[int, np.ndarray] = {}  # raw -> EMA feat (temp)
        self._next_id = 1
        self._last_out: list[dict] = []
        self._last_out_frame = 0
        self._sid_last_real: dict[int, int] = {}
        self._sid_last_embed_frame: dict[int, int] = {}
        self._last_identity_audit_frame = -10**9
        self._gallery_latest_at: dict[int, int] = {}
        self._gallery_first_saved: set[int] = set()
        self._gallery_best_area: dict[int, int] = {}
        self._gallery_best: dict[int, tuple[float, np.ndarray]] = {}
        self._gallery_first_wh: dict[int, tuple[int, int]] = {}
        self._gallery_first_feat: dict[int, np.ndarray] = {}
        self._gallery_first_color: dict[int, np.ndarray] = {}
        self._sid_exit_border: dict[int, bool] = {}
        self._sid_born_frame: dict[int, int] = {}
        self._unbind_votes: dict[int, tuple[str, int]] = {}
        self._unknown_raw_pending: dict[int, tuple[int, int]] = {}
        self._temp_dir: Path | None = None
        if self.gallery_dir is not None:
            if self.gallery_dir.exists():
                shutil.rmtree(self.gallery_dir)
            self.gallery_dir.mkdir(parents=True, exist_ok=True)
            self._temp_dir = self.gallery_dir / "temp"
            self._temp_dir.mkdir(parents=True, exist_ok=True)

    def _expire(self, frame_idx: int) -> None:
        stale_raw = [
            raw
            for raw, last in self._raw_last_frame.items()
            if frame_idx - int(last) > self.max_gap_frames
        ]
        for raw in stale_raw:
            self._raw_hits.pop(raw, None)
            self._raw_last_frame.pop(raw, None)
            self._raw_to_stable.pop(raw, None)
            self._raw_miss.pop(raw, None)
            self._pending_feat.pop(raw, None)

    def _alloc_stable_id(self) -> int:
        sid = self._next_id
        self._next_id += 1
        return sid

    def _freeze_other_galleries(
        self, new_sid: int, feat: np.ndarray | None
    ) -> None:
        """Strip the new outfit look from older IDs and freeze further enrollment.

        Only safe after a *long* leave + real outfit change. Must not run on
        pillar-occlusion mints (that splits one person into ID1/ID2).
        """
        if feat is None:
            return
        for sid, meta in list(self._stable.items()):
            if sid == new_sid:
                continue
            protos = list(self._proto_list(meta))
            if not protos:
                continue
            kept = [
                p for p in protos if self._appear_sim(feat, p) < self.appear_thresh
            ]
            if len(kept) == len(protos):
                continue
            if not kept:
                kept = [protos[0]]  # keep original enrollment look
            meta["feats"] = kept
            meta["feat"] = kept[0]
            meta["outfit_frozen"] = True
            self._stable[sid] = meta

    def _gallery_update_allowed(self, sid: int, feat: np.ndarray | None) -> bool:
        """Block writing my crop into a classmate's gallery (and vice versa)."""
        if feat is None or sid not in self._stable:
            return False
        meta = self._stable[sid]
        if not self._proto_list(meta):
            return True  # brand-new ID, first enrollment
        own = self._best_proto_sim(feat, meta)
        # Photos need a stronger match than tracking reuse.
        photo_bar = self.appear_thresh + 0.10
        if own < photo_bar:
            # Clearly closer to another identity → do not pollute this gallery.
            for other, other_meta in self._stable.items():
                if other == sid:
                    continue
                other_sim = self._best_proto_sim(feat, other_meta)
                if other_sim >= own + 0.04:
                    return False
            return False
        for other, other_meta in self._stable.items():
            if other == sid:
                continue
            other_sim = self._best_proto_sim(feat, other_meta)
            if other_sim >= own + 0.05 and other_sim >= photo_bar:
                return False
        return True

    @staticmethod
    def _crop_has_split_outfits(crop: np.ndarray) -> bool:
        """Heuristic: left/right halves look like two different shirts."""
        h, w = crop.shape[:2]
        if w < 48 or h < 64:
            return False
        left = crop[:, : max(1, w // 3)]
        right = crop[:, w - max(1, w // 3) :]
        ml = left.reshape(-1, 3).mean(axis=0)
        mr = right.reshape(-1, 3).mean(axis=0)
        # Strong chromatic split across the crop → mixed people at the edge.
        return float(np.linalg.norm(ml.astype(np.float32) - mr.astype(np.float32))) >= 35.0

    def _preferred_sid_for_feat(
        self,
        feat: np.ndarray | None,
        current_sid: int,
        blocked_sids: set[int],
    ) -> int | None:
        """If crop clearly matches another free gallery ID, return that sid."""
        if feat is None or current_sid not in self._stable:
            return None
        own = self._best_proto_sim(feat, self._stable[current_sid])
        best: tuple[float, int] | None = None
        for other, meta in self._stable.items():
            if other == current_sid or other in blocked_sids:
                continue
            if not self._proto_list(meta):
                continue
            sim = self._best_proto_sim(feat, meta)
            if sim >= self.appear_thresh and sim >= own + 0.06:
                if best is None or sim > best[0]:
                    best = (sim, other)
        return None if best is None else best[1]

    def _merge_duplicate(
        self,
        world: tuple[float, float] | None,
        feat: np.ndarray | None,
        used_sids: set[int],
        assigned_world: dict[int, tuple[float, float]],
        xyxy: tuple[int, int, int, int] | None = None,
        assigned_boxes: dict[int, tuple[int, int, int, int]] | None = None,
    ) -> int | None:
        """Collapse a second YOLO box of the *same* person only.

        Must NOT merge two nearby people (adjacent desks). Prefer overlapping
        boxes (edge split); otherwise require nearly the same foot point.
        """
        if xyxy is not None and assigned_boxes:
            overlap = self._overlap_sid(xyxy, assigned_boxes)
            if overlap is not None:
                return overlap
        if world is None:
            return None
        best: tuple[float, int] | None = None
        for sid in used_sids:
            meta = self._stable.get(sid)
            anchor = assigned_world.get(sid)
            if anchor is None and meta is not None:
                anchor = (float(meta["wx"]), float(meta["wy"]))
            if anchor is None:
                continue
            dist = self._dist(world, anchor)
            # ~one grid cell; two standing classmates are usually farther.
            if dist > 90.0:
                continue
            if dist <= 50.0:
                score = dist
            else:
                # 50–90cm: only if appearance also agrees (occlusion double-box).
                if feat is None or meta is None:
                    continue
                sim = self._best_proto_sim(feat, meta)
                if sim < self.soft_appear_thresh:
                    continue
                score = dist
            if best is None or score < best[0]:
                best = (score, sid)
        return None if best is None else best[1]

    def _overlap_sid(
        self,
        xyxy: tuple[int, int, int, int],
        assigned_boxes: dict[int, tuple[int, int, int, int]],
    ) -> int | None:
        best: int | None = None
        for sid, box in assigned_boxes.items():
            if not self._boxes_same_person(xyxy, box):
                continue
            if best is None or sid < best:
                best = sid
        return best

    def _short_gap_recover(
        self,
        world: tuple[float, float] | None,
        frame_idx: int,
        used_sids: set[int],
        feat: np.ndarray | None = None,
        color_feat: np.ndarray | None = None,
    ) -> int | None:
        """Recover ID after brief track loss (pillar occludes partial bbox)."""
        if world is None:
            return None
        recover_frames = max(int(self.sticky_frames), int(self.fps * 5))
        best: tuple[int, float, int] | None = None  # gap, dist, sid
        for sid, meta in self._stable.items():
            if sid in used_sids:
                continue
            gap = frame_idx - int(meta["frame"])
            if gap < 0 or gap > recover_frames:
                continue
            dist = self._dist(world, (float(meta["wx"]), float(meta["wy"])))
            # Foot point jumps a lot when half-hidden behind a pillar.
            limit = max(self._reach_limit_cm(gap), 350.0)
            if dist > limit:
                continue
            if self._appearance_forbids_reuse(sid, feat, color_feat):
                continue
            cand = (gap, dist, sid)
            if best is None or cand < best:
                best = cand
        return None if best is None else best[2]

    def _reuse_free_sid(
        self,
        world: tuple[float, float] | None,
        frame_idx: int,
        used_sids: set[int],
        feat: np.ndarray | None = None,
        color_feat: np.ndarray | None = None,
    ) -> int | None:
        """Reuse a recently-seen free ID by desk location.

        Appearance is a veto, not just a tie-break: a vacant slot is not
        reclaimed when the crop is clearly a different person.
        """
        recover_frames = max(int(self.sticky_frames), int(self.fps * 8))
        best: tuple[float, float, int] | None = None  # dist, -sim, sid
        for sid, meta in self._stable.items():
            if sid in used_sids:
                continue
            gap = frame_idx - int(meta["frame"])
            if gap < 0 or gap > recover_frames:
                continue
            if world is None:
                dist = 9999.0
            else:
                dist = self._dist(world, (float(meta["wx"]), float(meta["wy"])))
            limit = max(self._reach_limit_cm(gap), 250.0)
            if world is not None and dist > limit:
                continue
            if self._appearance_forbids_reuse(sid, feat, color_feat):
                continue
            sim = self._best_proto_sim(feat, meta) if feat is not None else 0.0
            cand = (dist, -sim, sid)
            if best is None or cand < best:
                best = cand
        return None if best is None else best[2]

    def _appearance_forbids_reuse(
        self,
        sid: int,
        feat: np.ndarray | None,
        color_feat: np.ndarray | None = None,
    ) -> bool:
        """True when this crop is clearly not that vacant ID."""
        meta = self._stable.get(sid)
        if meta is None or feat is None:
            return False
        sim = self._best_proto_sim(feat, meta)
        if sim < self.appear_thresh * 0.55:
            return True
        enrolled = self._gallery_first_color.get(sid)
        if (
            enrolled is not None
            and color_feat is not None
            and sim < self.appear_thresh
            and self._appear_sim(color_feat, enrolled) < 0.40
        ):
            return True
        return False

    def _det_color_feat(
        self, frame: np.ndarray | None, det: dict
    ) -> np.ndarray | None:
        if frame is None:
            return None
        xyxy = det.get("xyxy")
        if xyxy is None:
            return None
        return self._color_feat_from_crop(self._crop_person(frame, xyxy))

    def _recent_occluded_count(self, frame_idx: int) -> int:
        """Recent IDs last seen inside the room (not walking out the edge)."""
        window = max(int(self.sticky_frames), int(self.fps * 8))
        n = 0
        for sid, meta in self._stable.items():
            gap = frame_idx - int(meta["frame"])
            if gap < 0 or gap > window:
                continue
            if self._sid_exit_border.get(sid):
                continue
            n += 1
        return n

    def _should_refuse_mint(self, frame_idx: int, n_dets: int) -> bool:
        """Block a new ID while an interior person is only briefly missing.

        A real leave (last box on the image edge) does not count, so a
        different person walking in after ID1 left can still mint ID2.
        """
        n_occ = self._recent_occluded_count(frame_idx)
        return n_occ > 0 and n_occ >= n_dets

    @staticmethod
    def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)

    @staticmethod
    def appearance_feat(
        frame: np.ndarray,
        xyxy: tuple[int, int, int, int],
        encoder=None,
    ) -> np.ndarray | None:
        if encoder is not None:
            return encoder.embed_xyxy(frame, xyxy)
        # Same full YOLO person box as OSNet / gallery (not a torso-only crop).
        parsed = StableIdMapper._box_wh(frame, xyxy)
        if parsed is None:
            return None
        x1, y1, x2, y2, _bw, _bh = parsed
        crop = frame[y1:y2, x1:x2]
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
    def _color_feat_from_crop(crop: np.ndarray | None) -> np.ndarray | None:
        if crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        # In this seated camera view, the tracked body occupies the left/middle
        # of overlapped boxes; the right side often contains the person behind.
        torso = crop[int(0.18 * h) : int(0.70 * h), : int(0.58 * w)]
        if torso.size == 0:
            return None
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        feat = cv2.calcHist(
            [hsv], [0, 1], None, [24, 16], [0, 180, 0, 256]
        ).flatten().astype(np.float32)
        norm = float(np.linalg.norm(feat))
        return None if norm < 1e-6 else feat / norm

    @staticmethod
    def _appear_sim(a: np.ndarray | None, b: np.ndarray | None) -> float:
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b))

    def _blend_feat(
        self, old: np.ndarray | None, new: np.ndarray | None, alpha: float = 0.85
    ) -> np.ndarray | None:
        if new is None:
            return old
        if old is None:
            return new
        feat = alpha * old + (1.0 - alpha) * new
        n = float(np.linalg.norm(feat))
        return feat / n if n > 1e-6 else old

    def _proto_list(self, meta: dict) -> list[np.ndarray]:
        protos = meta.get("feats")
        if isinstance(protos, list) and protos:
            return [p for p in protos if p is not None]
        feat = meta.get("feat")
        return [feat] if feat is not None else []

    def _best_proto_sim(self, feat: np.ndarray | None, meta: dict) -> float:
        if feat is None:
            return 0.0
        best = 0.0
        for p in self._proto_list(meta):
            best = max(best, self._appear_sim(feat, p))
        return best

    def _update_prototypes(
        self,
        meta: dict,
        new_feat: np.ndarray | None,
        allow_new: bool = True,
    ) -> tuple[dict, int | None, bool]:
        """EMA nearest prototype, or append a new look on clothing change.

        Returns (meta, proto_index_to_snapshot, is_new_or_replaced).
        When ``allow_new`` is False (gallery rematch), never enroll a divergent
        stranger look under this ID — only EMA an existing close match.
        """
        if new_feat is None:
            return meta, None, False
        protos = list(self._proto_list(meta))
        if not protos:
            meta["feats"] = [new_feat]
            meta["feat"] = new_feat
            return meta, 0, True

        sims = [self._appear_sim(new_feat, p) for p in protos]
        best_i = int(np.argmax(sims))
        best_sim = float(sims[best_i])

        if best_sim >= self.proto_new_thresh:
            protos[best_i] = self._blend_feat(protos[best_i], new_feat, alpha=0.85)
            snap_i, is_new = best_i, False
        elif allow_new:
            # Continuous same track + large look shift (jacket off): add proto.
            unlimited = self.max_prototypes <= 0
            if unlimited or len(protos) < self.max_prototypes:
                protos.append(new_feat)
                snap_i, is_new = len(protos) - 1, True
            else:
                worst_i = int(np.argmin(sims))
                protos[worst_i] = new_feat
                snap_i, is_new = worst_i, True
        else:
            # Rematch path: do not pollute gallery with a different person.
            protos[best_i] = self._blend_feat(protos[best_i], new_feat, alpha=0.95)
            snap_i, is_new = best_i, False

        meta["feats"] = protos
        meta["feat"] = protos[best_i] if best_sim >= self.proto_new_thresh else protos[-1]
        return meta, snap_i, is_new

    @staticmethod
    def _box_wh(
        frame: np.ndarray, xyxy: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int, int, int] | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw < 8 or bh < 16:
            return None
        return x1, y1, x2, y2, bw, bh

    @staticmethod
    def _box_iou(
        a: tuple[int, int, int, int], b: tuple[int, int, int, int]
    ) -> float:
        ax1, ay1, ax2, ay2 = [int(v) for v in a]
        bx1, by1, bx2, by2 = [int(v) for v in b]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / float(area_a + area_b - inter)

    @classmethod
    def _boxes_same_person(
        cls,
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> bool:
        """True when two boxes are a split / nested detection of one body."""
        iou = cls._box_iou(a, b)
        if iou >= 0.40:
            return True
        ax1, ay1, ax2, ay2 = [int(v) for v in a]
        bx1, by1, bx2, by2 = [int(v) for v in b]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter <= 0:
            return False
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        # Nested / truncated fragment inside a larger person box.
        return inter / float(min(area_a, area_b)) >= 0.60

    @classmethod
    def _gallery_conflicts_others(
        cls,
        xyxy: tuple[int, int, int, int],
        others: list[tuple[int, int, int, int]] | None,
    ) -> bool:
        """True when another detected person can contaminate this crop."""
        if not others:
            return False
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        area = max(1, (x2 - x1) * (y2 - y1))
        for ob in others:
            iou = cls._box_iou(xyxy, ob)
            if iou >= 0.35:
                return True
            ox1, oy1, ox2, oy2 = [int(v) for v in ob]
            ix1, iy1 = max(x1, ox1), max(y1, oy1)
            ix2, iy2 = min(x2, ox2), min(y2, oy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            oarea = max(1, (ox2 - ox1) * (oy2 - oy1))
            smaller = min(area, oarea)
            # Even partial overlap can put a classmate's shirt/face into OSNet.
            if inter / float(smaller) >= 0.65:
                return True
        return False

    @classmethod
    def _gallery_person_score(
        cls,
        frame: np.ndarray,
        xyxy: tuple[int, int, int, int],
        conf: float | None = None,
        others: list[tuple[int, int, int, int]] | None = None,
    ) -> float:
        """Prefer complete, in-frame person boxes for gallery photos."""
        parsed = cls._box_wh(frame, xyxy)
        if parsed is None:
            return 0.0
        x1, y1, x2, y2, bw, bh = parsed
        fh, fw = frame.shape[:2]
        aspect = bh / float(max(1, bw))
        # Too skinny OR too wide (often two people / wrong mega-box).
        if aspect > 3.6 or aspect < 1.35:
            return 0.0
        # Absolute mega-boxes still rejected; near-camera alone shots can be
        # larger because _tighten_gallery_box will center-crop the torso.
        if bw > int(0.24 * fw) or bh > int(0.85 * fh):
            return 0.0
        if bw < max(70, int(0.04 * fw)):
            return 0.0
        if (bw * bh) > int(0.18 * fw * fh):
            return 0.0
        if cls._gallery_conflicts_others(xyxy, others):
            return 0.0
        # Side/edge scraps must not overwrite latest.jpg.
        if x1 <= 2 or x2 >= fw - 3:
            return 0.0
        if y1 <= 2 and bh < 0.35 * fh:
            return 0.0
        # Prefer mid-size boxes; huge near-camera boxes get a mild penalty
        # (tight crop still saves a usable photo).
        area_r = (bw * bh) / float(max(1, fw * fh))
        size_pen = 1.0 if area_r <= 0.06 else max(0.35, 1.0 - (area_r - 0.06) * 4.0)
        c = 1.0 if conf is None else max(0.05, float(conf))
        aspect_pen = 1.0 / (1.0 + abs(aspect - 2.2))
        return area_r * 1000.0 * c * aspect_pen * size_pen

    @classmethod
    def _is_full_person_box(
        cls,
        frame: np.ndarray,
        xyxy: tuple[int, int, int, int],
        conf: float | None = None,
        others: list[tuple[int, int, int, int]] | None = None,
    ) -> bool:
        return cls._gallery_person_score(frame, xyxy, conf, others=others) > 0.0

    @classmethod
    def _expand_person_box(
        cls, frame: np.ndarray, xyxy: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int] | None:
        """Light pad around YOLO box; widen only if the box is too skinny."""
        parsed = cls._box_wh(frame, xyxy)
        if parsed is None:
            return None
        x1, y1, x2, y2, bw, bh = parsed
        fh, fw = frame.shape[:2]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        aspect = bh / float(max(1, bw))
        pad_x = 0.03 * bw
        pad_y = 0.04 * bh
        need_w = bw + 2 * pad_x
        need_h = bh + 2 * pad_y
        if aspect > 2.8:
            need_w = max(need_w, bh / 2.6)
        nx1 = int(round(cx - 0.5 * need_w))
        nx2 = int(round(cx + 0.5 * need_w))
        ny1 = int(round(cy - 0.5 * need_h))
        ny2 = int(round(cy + 0.5 * need_h))
        nx1 = max(0, nx1)
        ny1 = max(0, ny1)
        nx2 = min(fw - 1, nx2)
        ny2 = min(fh - 1, ny2)
        if nx2 - nx1 < 40 or ny2 - ny1 < 80:
            return None
        return nx1, ny1, nx2, ny2

    @classmethod
    def _tighten_gallery_box(
        cls, frame: np.ndarray, xyxy: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int] | None:
        """For near-camera mega-boxes, keep a center torso strip.

        Far classmates often sit in the upper/side margins of a close box; a
        tight center crop keeps first/latest.jpg as one person.
        """
        box = cls._expand_person_box(frame, xyxy)
        if box is None:
            return None
        x1, y1, x2, y2 = box
        fh, fw = frame.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        mega = bw > int(0.11 * fw) or (bw * bh) > int(0.07 * fw * fh)
        if mega:
            cx = 0.5 * (x1 + x2)
            keep_w = max(72, min(int(0.42 * bw), int(0.085 * fw)))
            x1 = int(round(cx - 0.5 * keep_w))
            x2 = int(round(cx + 0.5 * keep_w))
            # Drop upper band (distant people) and a little of the feet.
            y1 = y1 + int(0.18 * bh)
            y2 = y2 - int(0.06 * bh)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(fw - 1, x2)
        y2 = min(fh - 1, y2)
        if x2 - x1 < 40 or y2 - y1 < 80:
            return None
        return x1, y1, x2, y2

    @classmethod
    def _crop_person(
        cls, frame: np.ndarray, xyxy: tuple[int, int, int, int]
    ) -> np.ndarray | None:
        """Gallery crop: YOLO person box, tightened if near-camera mega-box."""
        box = cls._tighten_gallery_box(frame, xyxy)
        if box is None:
            return None
        x1, y1, x2, y2 = box
        return frame[y1:y2, x1:x2].copy()

    def _write_gallery_crop(self, sid: int, name: str, crop: np.ndarray) -> None:
        if self.gallery_dir is None:
            return
        folder = self.gallery_dir / f"ID{sid:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / name), crop)

    def _write_temp_compare(
        self,
        frame: np.ndarray | None,
        xyxy: tuple[int, int, int, int] | None,
        raw: int | None = None,
        tag: str = "current",
    ) -> None:
        """Save the crop currently being compared against past ID photos."""
        if self._temp_dir is None or frame is None or xyxy is None:
            return
        crop = self._crop_person(frame, xyxy)
        if crop is None:
            parsed = self._box_wh(frame, xyxy)
            if parsed is None:
                return
            x1, y1, x2, y2, _bw, _bh = parsed
            crop = frame[y1:y2, x1:x2].copy()
        cv2.imwrite(str(self._temp_dir / f"{tag}.jpg"), crop)
        if raw is not None:
            cv2.imwrite(str(self._temp_dir / f"raw_{int(raw)}.jpg"), crop)

    def _nearby_other_person(
        self,
        sid: int,
        world: tuple[float, float] | None,
        frame_idx: int,
        *,
        max_dist_cm: float = 160.0,
        max_age_frames: int | None = None,
    ) -> bool:
        """True if another ID was just seen close by (risky for clean gallery crops)."""
        if world is None:
            return False
        age = int(self.fps * 1.5) if max_age_frames is None else int(max_age_frames)
        wx, wy = float(world[0]), float(world[1])
        for other, meta in self._stable.items():
            if other == sid:
                continue
            if frame_idx - int(meta.get("frame", -10**9)) > age:
                continue
            if self._dist((wx, wy), (float(meta["wx"]), float(meta["wy"]))) <= max_dist_cm:
                return True
        return False

    def _other_id_recent(
        self, sid: int, frame_idx: int, *, max_age_frames: int | None = None
    ) -> bool:
        age = int(self.fps * 3.0) if max_age_frames is None else int(max_age_frames)
        for other, meta in self._stable.items():
            if other == sid:
                continue
            if frame_idx - int(meta.get("frame", -10**9)) <= age:
                return True
        return False

    def _matches_enrollment(
        self, sid: int, feat: np.ndarray | None, *, margin: float = 0.05
    ) -> bool:
        """latest.jpg must still look like first.jpg (unless explicit new proto)."""
        if feat is None:
            return False
        first = self._gallery_first_feat.get(sid)
        if first is None:
            return True
        return self._appear_sim(feat, first) >= (self.appear_thresh + margin)

    def _crop_looks_like_one_person(
        self,
        sid: int,
        crop: np.ndarray,
        frame: np.ndarray,
        xyxy: tuple[int, int, int, int],
        conf: float | None,
        others: list[tuple[int, int, int, int]] | None = None,
    ) -> bool:
        """Reject mega-boxes that swallowed a second person into this ID's photo."""
        if self._gallery_person_score(frame, xyxy, conf, others=others) <= 0.0:
            return False
        h, w = crop.shape[:2]
        first_wh = self._gallery_first_wh.get(sid)
        if first_wh is not None:
            fw0, fh0 = first_wh
            # New crop much wider/larger than enrollment → likely mixed people.
            if w > int(1.45 * fw0) or (w * h) > int(1.9 * fw0 * fh0):
                return False
        # Extreme motion blur → unreliable Re-ID / gallery pollution.
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < 40.0:
            return False
        if self._crop_has_split_outfits(crop):
            return False
        return True

    def _update_gallery_best(
        self,
        sid: int,
        frame: np.ndarray | None,
        xyxy: tuple[int, int, int, int] | None,
        conf: float | None,
        frame_idx: int,
        *,
        also_proto: int | None = None,
        capture_first: bool = False,
        others: list[tuple[int, int, int, int]] | None = None,
        feat: np.ndarray | None = None,
    ) -> bool:
        """Gallery dump.

        ``first.jpg`` = the frame when this ID was first marked (never overwritten).
        ``latest.jpg`` = only a clean single-person crop that still matches this ID.
        Returns True if a gallery image was written.
        """
        if self.gallery_dir is None or frame is None or xyxy is None:
            return False
        if self._gallery_conflicts_others(xyxy, others):
            return False
        crop = self._crop_person(frame, xyxy)
        if crop is None and capture_first:
            parsed = self._box_wh(frame, xyxy)
            if parsed is not None:
                x1, y1, x2, y2, _bw, _bh = parsed
                crop = frame[y1:y2, x1:x2].copy()
        if crop is None:
            return False
        score = self._gallery_person_score(frame, xyxy, conf, others=others)

        if capture_first and sid not in self._gallery_first_saved:
            self._write_gallery_crop(sid, "first.jpg", crop)
            self._write_gallery_crop(sid, "proto_0.jpg", crop)
            self._write_gallery_crop(sid, "latest.jpg", crop)
            self._gallery_first_saved.add(sid)
            self._gallery_latest_at[sid] = frame_idx
            self._gallery_first_wh[sid] = (int(crop.shape[1]), int(crop.shape[0]))
            self._gallery_best[sid] = (score, crop)
            self._gallery_best_area[sid] = int(crop.shape[0] * crop.shape[1])
            if feat is not None:
                self._gallery_first_feat[sid] = feat.copy()
            color_feat = self._color_feat_from_crop(crop)
            if color_feat is not None:
                self._gallery_first_color[sid] = color_feat
            return True

        if sid not in self._gallery_first_saved:
            return False

        if also_proto is not None and also_proto > 0:
            self._write_gallery_crop(sid, f"proto_{also_proto}.jpg", crop)
            self._write_gallery_crop(sid, "latest.jpg", crop)
            self._gallery_latest_at[sid] = frame_idx
            if score > 0.0:
                prev = self._gallery_best.get(sid)
                if prev is None or score >= float(prev[0]) * 1.02:
                    self._gallery_best[sid] = (score, crop)
                    self._gallery_best_area[sid] = int(
                        crop.shape[0] * crop.shape[1]
                    )
            return True
        if score <= 0.0:
            due = (
                frame_idx - self._gallery_latest_at.get(sid, -10**9)
                >= self.gallery_latest_every
            )
            if due:
                self._write_gallery_crop(sid, "latest.jpg", crop)
                self._gallery_latest_at[sid] = frame_idx
                return True
            return False

        prev = self._gallery_best.get(sid)
        best_score = 0.0 if prev is None else float(prev[0])
        improved = prev is None or score >= best_score * 1.02
        good_enough = score >= max(best_score * 0.92, 1e-6)
        if not improved and not good_enough:
            return False
        if improved:
            self._gallery_best[sid] = (score, crop)
            self._gallery_best_area[sid] = int(crop.shape[0] * crop.shape[1])
        due = (
            frame_idx - self._gallery_latest_at.get(sid, -10**9)
            >= self.gallery_latest_every
        )
        if improved or (due and good_enough):
            self._write_gallery_crop(sid, "latest.jpg", crop)
            self._gallery_latest_at[sid] = frame_idx
            return True
        return False

    def _save_gallery_image(
        self,
        sid: int,
        name: str,
        frame: np.ndarray | None,
        xyxy: tuple[int, int, int, int] | None,
        *,
        require_full_body: bool = True,
        conf: float | None = None,
        others: list[tuple[int, int, int, int]] | None = None,
    ) -> bool:
        if self.gallery_dir is None or frame is None or xyxy is None:
            return False
        if require_full_body and not self._is_full_person_box(
            frame, xyxy, conf, others=others
        ):
            return False
        crop = self._crop_person(frame, xyxy)
        if crop is None:
            return False
        self._write_gallery_crop(sid, name, crop)
        return True

    def _reach_limit_cm(self, gap_frames: int) -> float:
        gap_s = max(0, gap_frames) / self.fps
        return min(self.max_dist_cm, self.max_speed_cm_s * gap_s + 80.0)

    def _gallery_hits(
        self, feat: np.ndarray | None, used_sids: set[int], thresh: float
    ) -> list[tuple[float, int]]:
        """Return [(sim, sid), ...] with max-over-prototypes sim >= thresh."""
        hits: list[tuple[float, int]] = []
        if feat is None:
            return hits
        for sid, meta in self._stable.items():
            if sid in used_sids:
                continue
            sim = self._best_proto_sim(feat, meta)
            if sim >= thresh:
                hits.append((sim, sid))
        hits.sort(key=lambda t: (-t[0], t[1]))
        return hits

    def _pick_gallery_id(
        self, feat: np.ndarray | None, used_sids: set[int], thresh: float
    ) -> tuple[int | None, float]:
        hits = self._gallery_hits(feat, used_sids, thresh)
        if not hits:
            return None, -1.0
        best_sim = hits[0][0]
        near = [sid for sim, sid in hits if sim >= best_sim - 0.03]
        # Prefer the ID seen most recently (clothing-change ID3 over dormant ID1).
        sid = max(near, key=lambda s: int(self._stable[s]["frame"]))
        sim = max(s for s, i in hits if i == sid)
        return sid, sim

    def _recent_gap_frames(self) -> int:
        return max(int(self.sticky_frames * 2), int(self.fps * 4))

    def _rematch_allowed(
        self,
        sid: int,
        feat: np.ndarray | None,
        world: tuple[float, float] | None,
        frame_idx: int,
        min_sim: float | None = None,
        last_chance: bool = False,
    ) -> bool:
        """Allow same-person re-entry; block two-person merges.

        - Similarity: ``appear_thresh`` normally; after a longer leave, soft bar.
        - Spatial: if that ID was seen *recently* but the new foot is too far,
          reject (different desk / different person). Long leave ignores distance.
        """
        if sid not in self._stable:
            return False
        meta = self._stable[sid]
        sim = self._best_proto_sim(feat, meta)
        gap = frame_idx - int(meta["frame"])
        recent = self._recent_gap_frames()
        if min_sim is not None:
            need = min_sim
        elif gap > recent:
            need = (
                self.appear_thresh * 0.70 if last_chance else self.soft_appear_thresh
            )
        else:
            need = self.appear_thresh
        if sim < need:
            return False
        if world is None:
            return True
        if gap <= recent:
            dist = self._dist(world, (float(meta["wx"]), float(meta["wy"])))
            # Tight spatial gate: nearby classmate must not steal this ID.
            limit = min(max(self._reach_limit_cm(gap), 100.0), 140.0)
            if dist > limit:
                return False
        return True

    def _fix_id_swaps(
        self,
        work: list[dict],
        assigned: dict[int, int],
        from_raw: set[int],
        assigned_world: dict[int, tuple[float, float]],
        feat_at,
    ) -> None:
        """If two people were crossed, reassign IDs to maximize appearance match."""
        import itertools

        # A confirmed newcomer can receive a SID earlier in this frame; its
        # stable metadata/prototypes are created later in the output pass.
        # Swap auditing must only compare already-enrolled identities.
        idxs = [
            i
            for i, sid in assigned.items()
            if sid in self._stable and self._proto_list(self._stable[sid])
        ]
        if len(idxs) < 2 or len(idxs) > 4:
            return
        sids = [assigned[i] for i in idxs]
        feats = [feat_at(i, force=True) for i in idxs]
        if any(f is None for f in feats):
            return

        def score_perm(perm: tuple[int, ...]) -> float:
            total = 0.0
            for a, b in enumerate(perm):
                total += self._best_proto_sim(feats[a], self._stable[sids[b]])
            return total

        base = score_perm(tuple(range(len(idxs))))
        best_perm = tuple(range(len(idxs)))
        best = base
        for perm in itertools.permutations(range(len(idxs))):
            sc = score_perm(perm)
            if sc > best + 0.12:
                best = sc
                best_perm = perm
        if best_perm == tuple(range(len(idxs))):
            return
        # Appearance-only swaps teleport seated classmates when clothes look alike.
        for a, b in enumerate(best_perm):
            if a == b:
                continue
            w = work[idxs[a]].get("world")
            if w is None:
                continue
            meta_old = self._stable.get(sids[a])
            meta_new = self._stable.get(sids[b])
            if meta_old is None or meta_new is None:
                continue
            dist_old = self._dist(w, (float(meta_old["wx"]), float(meta_old["wy"])))
            dist_new = self._dist(w, (float(meta_new["wx"]), float(meta_new["wy"])))
            if dist_old + 80.0 < dist_new:
                return
        # Apply swap.
        new_map = {idxs[a]: sids[b] for a, b in enumerate(best_perm)}
        assigned.clear()
        assigned.update(new_map)
        used = set(assigned.values())
        from_raw.clear()
        assigned_world.clear()
        for i, sid in assigned.items():
            raw = work[i].get("track_id")
            if raw is not None:
                self._raw_to_stable[int(raw)] = sid
                from_raw.add(i)
            w = work[i].get("world")
            if w is not None:
                assigned_world[sid] = (float(w[0]), float(w[1]))
        # silence unused
        _ = used

    def _audit_color_bindings(
        self,
        work: list[dict],
        assigned: dict[int, int],
        used_sids: set[int],
        assigned_world: dict[int, tuple[float, float]],
        frame: np.ndarray | None,
        drop_unknown: bool = False,
    ) -> None:
        """Correct obvious raw-track swaps using clean enrollment clothing."""
        if frame is None or len(self._gallery_first_color) < 2:
            return
        colors: dict[int, np.ndarray] = {}
        for i in assigned:
            xyxy = work[i].get("xyxy")
            color = (
                self._color_feat_from_crop(self._crop_person(frame, xyxy))
                if xyxy is not None
                else None
            )
            if color is not None:
                colors[i] = color

        # Both IDs are occupied, but each crop matches the other's enrollment.
        indices = list(assigned)
        for pos, i in enumerate(indices):
            for j in indices[pos + 1 :]:
                sid_i, sid_j = assigned[i], assigned[j]
                ci, cj = colors.get(i), colors.get(j)
                ei = self._gallery_first_color.get(sid_i)
                ej = self._gallery_first_color.get(sid_j)
                if ci is None or cj is None or ei is None or ej is None:
                    continue
                own = self._appear_sim(ci, ei) + self._appear_sim(cj, ej)
                cross_i = self._appear_sim(ci, ej)
                cross_j = self._appear_sim(cj, ei)
                if (
                    min(cross_i, cross_j) < 0.45
                    or max(cross_i, cross_j) < 0.65
                    or cross_i + cross_j < own + 0.35
                ):
                    continue
                assigned[i], assigned[j] = sid_j, sid_i
                wi, wj = work[i].get("world"), work[j].get("world")
                if wi is not None:
                    assigned_world[sid_j] = (float(wi[0]), float(wi[1]))
                if wj is not None:
                    assigned_world[sid_i] = (float(wj[0]), float(wj[1]))
                for k, sid in ((i, sid_j), (j, sid_i)):
                    raw = work[k].get("track_id")
                    if raw is not None:
                        self._raw_to_stable[int(raw)] = sid

        for i, sid in list(assigned.items()):
            own_enrollment = self._gallery_first_color.get(sid)
            color = colors.get(i)
            if own_enrollment is None or color is None:
                continue
            own = self._appear_sim(color, own_enrollment)
            best: tuple[float, int] | None = None
            for other, enrollment in self._gallery_first_color.items():
                if other == sid or other in used_sids:
                    continue
                sim = self._appear_sim(color, enrollment)
                if best is None or sim > best[0]:
                    best = (sim, other)
            can_switch = (
                best is not None
                and own < 0.45
                and best[0] >= 0.65
                and best[0] >= own + 0.20
            )
            if not can_switch:
                if drop_unknown and own < 0.40:
                    assigned.pop(i, None)
                    used_sids.discard(sid)
                    assigned_world.pop(sid, None)
                    raw = work[i].get("track_id")
                    if raw is not None:
                        raw = int(raw)
                        self._raw_to_stable.pop(raw, None)
                        count, _last = self._unknown_raw_pending.get(raw, (0, -1))
                        self._unknown_raw_pending[raw] = (count + 1, self._raw_last_frame.get(raw, 0))
                continue
            assert best is not None
            other = best[1]
            used_sids.discard(sid)
            used_sids.add(other)
            assigned[i] = other
            assigned_world.pop(sid, None)
            world = work[i].get("world")
            if world is not None:
                assigned_world[other] = (float(world[0]), float(world[1]))
            raw = work[i].get("track_id")
            if raw is not None:
                self._raw_to_stable[int(raw)] = other

    def _audit_raw_bindings(
        self,
        work: list[dict],
        assigned: dict[int, int],
        used_sids: set[int],
        from_raw: set[int],
        assigned_world: dict[int, tuple[float, float]],
        feat_at,
        frame_idx: int,
    ) -> None:
        """Break ByteTrack links that clearly belong to another gallery ID."""
        for i in list(from_raw):
            sid = assigned.get(i)
            if sid is None or sid not in self._stable:
                continue
            feat = feat_at(i, force=True)
            if feat is None:
                continue
            meta = self._stable[sid]
            own = self._best_proto_sim(feat, meta)
            world = work[i].get("world")
            # Impossible jump in a few frames → tracker swapped bodies.
            if world is not None:
                dist = self._dist(world, (float(meta["wx"]), float(meta["wy"])))
                max_jump = max(160.0, self.max_speed_cm_s * (2.5 / self.fps) + 60.0)
                if dist > max_jump and own < self.appear_thresh + 0.05:
                    used_sids.discard(sid)
                    assigned.pop(i, None)
                    from_raw.discard(i)
                    assigned_world.pop(sid, None)
                    raw = work[i].get("track_id")
                    if raw is not None:
                        self._raw_to_stable.pop(int(raw), None)
                    continue
            best_other: tuple[float, int] | None = None
            for other, ometa in self._stable.items():
                if other == sid or other in used_sids:
                    continue
                osim = self._best_proto_sim(feat, ometa)
                if best_other is None or osim > best_other[0]:
                    best_other = (osim, other)
            if (
                best_other is not None
                and best_other[0] >= own + 0.14
                and best_other[0] >= self.appear_thresh + 0.05
                and own < self.soft_appear_thresh
            ):
                raw = work[i].get("track_id")
                key = int(raw) if raw is not None else (-1000 - i)
                votes = self._unbind_votes.get(key, ("other", 0))
                self._unbind_votes[key] = ("other", votes[1] + 1)
                if self._unbind_votes[key][1] < 3:
                    continue
                self._unbind_votes.pop(key, None)
                used_sids.discard(sid)
                assigned.pop(i, None)
                from_raw.discard(i)
                assigned_world.pop(sid, None)
                if raw is not None:
                    self._raw_to_stable.pop(int(raw), None)
            else:
                raw = work[i].get("track_id")
                if raw is not None:
                    self._unbind_votes.pop(int(raw), None)

    def _select_rematch(
        self,
        feat: np.ndarray | None,
        world: tuple[float, float] | None,
        frame_idx: int,
        used_sids: set[int],
        last_chance: bool = False,
    ) -> tuple[int | None, float]:
        """Best gallery rematch: appearance + spatial nearness (not newest ID)."""
        best: tuple[float, float, float, int] | None = None
        # score, sim, -dist, -sid  (older ID wins exact ties)
        for sid, meta in self._stable.items():
            if sid in used_sids:
                continue
            if not self._rematch_allowed(
                sid, feat, world, frame_idx, last_chance=last_chance
            ):
                continue
            sim = self._best_proto_sim(feat, meta)
            dist = 0.0
            if world is not None:
                dist = self._dist(world, (float(meta["wx"]), float(meta["wy"])))
            # Prefer same desk / nearby foot. Do NOT prefer globally newest ID
            # (that caused ID1↔ID2↔ID3 thrashing in recordings).
            score = float(sim)
            if world is not None:
                if dist <= 80.0:
                    score += 0.06
                elif dist <= 150.0:
                    score += 0.03
                else:
                    score -= min(0.12, (dist - 150.0) / 2000.0)
            cand = (score, sim, -dist, -sid)
            if best is None or cand > best:
                best = cand
        if best is None:
            return None, -1.0
        return -best[3], best[1]

    def _retire_young_sid(self, lose_sid: int, win_sid: int, frame_idx: int) -> None:
        """Drop a freshly minted split-ID so leave/re-enter keeps ID1."""
        if lose_sid == win_sid:
            return
        born = self._sid_born_frame.get(lose_sid)
        age = frame_idx - int(born) if born is not None else 10**9
        if age > int(max(self.fps * 4, 80)):
            return
        self._stable.pop(lose_sid, None)
        self._sid_born_frame.pop(lose_sid, None)
        self._sid_last_embed_frame.pop(lose_sid, None)
        self._gallery_latest_at.pop(lose_sid, None)
        self._gallery_first_saved.discard(lose_sid)
        self._gallery_best_area.pop(lose_sid, None)
        self._gallery_best.pop(lose_sid, None)
        self._gallery_first_wh.pop(lose_sid, None)
        self._gallery_first_feat.pop(lose_sid, None)
        self._gallery_first_color.pop(lose_sid, None)
        self._sid_exit_border.pop(lose_sid, None)
        for raw, sid in list(self._raw_to_stable.items()):
            if sid == lose_sid:
                self._raw_to_stable[raw] = win_sid
        if lose_sid == self._next_id - 1:
            self._next_id = lose_sid

    def _collapse_split_outputs(
        self, out: list[dict], frame_idx: int
    ) -> list[dict]:
        if len(out) < 2:
            return out
        keep = [True] * len(out)
        for i in range(len(out)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(out)):
                if not keep[j]:
                    continue
                if not self._boxes_same_person(out[i]["xyxy"], out[j]["xyxy"]):
                    continue
                si = int(out[i]["track_id"])
                sj = int(out[j]["track_id"])
                if si == sj:
                    keep[j] = False
                    raw = out[j].get("raw_track_id")
                    if raw is not None:
                        self._raw_to_stable[int(raw)] = si
                    continue
                color_i = self._gallery_first_color.get(si)
                color_j = self._gallery_first_color.get(sj)
                if (
                    color_i is not None
                    and color_j is not None
                    and self._appear_sim(color_i, color_j) < 0.45
                ):
                    # Adjacent classmates can have nested boxes while seated.
                    # Distinct clean enrollment colors prove they are not a
                    # split detection of one body.
                    continue
                loser, winner = (j, i) if si < sj else (i, j)
                keep[loser] = False
                win_sid = int(out[winner]["track_id"])
                lose_sid = int(out[loser]["track_id"])
                raw = out[loser].get("raw_track_id")
                if raw is not None:
                    self._raw_to_stable[int(raw)] = win_sid
                self._retire_young_sid(lose_sid, win_sid, frame_idx)
        return [d for d, k in zip(out, keep) if k]

    def _dump_review(
        self,
        out: list[dict],
        frame: np.ndarray | None,
        frame_idx: int,
    ) -> None:
        if self.review is None or frame is None:
            return
        for d in out:
            sid = d.get("track_id")
            xyxy = d.get("xyxy")
            if sid is None or xyxy is None:
                continue
            self.review.maybe_save(int(sid), frame, xyxy, frame_idx)

    @staticmethod
    def _box_at_border(
        det: dict, frame: np.ndarray | None, frac: float = 0.04
    ) -> bool:
        """True when the box is clipped to the image edge (person walking out)."""
        if frame is None:
            return False
        xyxy = det.get("xyxy")
        if xyxy is None:
            return False
        h, w = frame.shape[:2]
        m = max(16, int(frac * min(w, h)))
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        return x1 <= m or y1 <= m or x2 >= w - m or y2 >= h - m

    def _coasted_empty_out(
        self, frame_idx: int, frame: np.ndarray | None
    ) -> list[dict]:
        """Hold interior boxes briefly on a total miss; never after an exit."""
        if (
            not self._last_out
            or self.coast_frames <= 0
            or frame_idx - self._last_out_frame > self.coast_frames
        ):
            return []
        keep = [
            dict(d)
            for d in self._last_out
            if not self._box_at_border(d, frame)
        ]
        return keep

    def _hold_missing_interior(
        self,
        out: list[dict],
        frame_idx: int,
        frame: np.ndarray | None,
    ) -> list[dict]:
        """Keep a recently seen ID if YOLO missed them this frame.

        Does not assume how many people are present. Boxes at the image edge
        (walking out) are not held — those must disappear immediately.
        """
        hold = max(int(self.coast_frames), int(self.fps * 1.2))
        if hold <= 0:
            return out
        have = {
            int(d["track_id"])
            for d in out
            if d.get("track_id") is not None
        }
        extra: list[dict] = []
        for prev in self._last_out:
            sid = prev.get("track_id")
            if sid is None:
                continue
            sid = int(sid)
            if sid in have:
                continue
            last = self._sid_last_real.get(sid, self._last_out_frame)
            if frame_idx - int(last) > hold:
                continue
            if self._box_at_border(prev, frame):
                continue
            extra.append(dict(prev))
            have.add(sid)
        return out + extra

    def apply(
        self,
        dets: list[dict],
        frame_idx: int,
        frame: np.ndarray | None = None,
    ) -> list[dict]:
        self._expire(frame_idx)
        if not dets:
            return self._coasted_empty_out(frame_idx, frame)

        work = list(dets)
        if self.single_person and len(work) > 1:
            if self._stable:
                _sid, meta = max(self._stable.items(), key=lambda kv: kv[1]["frame"])
                anchor = (float(meta["wx"]), float(meta["wy"]))
                work = [
                    min(work, key=lambda d: self._dist(d.get("world", (1e9, 1e9)), anchor))
                ]
            else:
                work = [max(work, key=lambda d: float(d.get("conf", 0.0)))]

        # Lazy Re-ID: only embed when a detection is not already bound by raw ID.
        # OSNet on CPU is ~100ms/crop — skipping continuous tracks cuts lag a lot.
        feats: list[np.ndarray | None] = [None] * len(work)

        def feat_at(i: int, force: bool = False) -> np.ndarray | None:
            if feats[i] is not None:
                return feats[i]
            if frame is None:
                return None
            if not force and i in from_raw and (frame_idx % 10) != 0:
                return None
            feats[i] = self.appearance_feat(
                frame, work[i]["xyxy"], encoder=self.encoder
            )
            return feats[i]

        def color_at(i: int) -> np.ndarray | None:
            return self._det_color_feat(frame, work[i])

        assigned: dict[int, int] = {}
        used_sids: set[int] = set()
        from_raw: set[int] = set()  # continuous ByteTrack → may learn new looks
        assigned_world: dict[int, tuple[float, float]] = {}
        pending_unknown: set[int] = set()

        # 1) Continuous YOLO/ByteTrack box → keep the same stable ID
        #    (clothes change later records a photo, does not mint).
        for i, det in enumerate(work):
            raw = det.get("track_id")
            if raw is None:
                continue
            raw = int(raw)
            pending = self._unknown_raw_pending.get(raw)
            if pending is not None:
                count = pending[0] + 1
                if count < max(3, self.min_hits // 4):
                    self._unknown_raw_pending[raw] = (count, frame_idx)
                    pending_unknown.add(i)
                    continue
                self._unknown_raw_pending.pop(raw, None)
                sid = self._alloc_stable_id()
                assigned[i] = sid
                used_sids.add(sid)
                self._raw_to_stable[raw] = sid
                w = det.get("world")
                if w is not None:
                    assigned_world[sid] = (float(w[0]), float(w[1]))
                continue
            sid = self._raw_to_stable.get(raw)
            if sid is None or sid not in self._stable or sid in used_sids:
                continue
            assigned[i] = sid
            used_sids.add(sid)
            from_raw.add(i)
            w = det.get("world")
            if w is not None:
                assigned_world[sid] = (float(w[0]), float(w[1]))

        self._audit_color_bindings(
            work, assigned, used_sids, assigned_world, frame
        )

        # 1b) Audit body swaps periodically. Running OSNet over every person
        # on every detection cannot keep up on the CPU-only setup.
        identity_audit_due = (
            frame_idx - self._last_identity_audit_frame >= max(1, int(self.fps))
        )
        if len(work) >= 2 and identity_audit_due:
            self._audit_raw_bindings(
                work,
                assigned,
                used_sids,
                from_raw,
                assigned_world,
                feat_at,
                frame_idx,
            )
            self._last_identity_audit_frame = frame_idx

        # 2) Short-gap spatial recovery FIRST (stop ID thrash on brief drops).
        for i in range(len(work)):
            if i in assigned or i in pending_unknown:
                continue
            sid = self._short_gap_recover(
                work[i].get("world"),
                frame_idx,
                used_sids,
                feat_at(i, force=True),
                color_feat=color_at(i),
            )
            if sid is None:
                continue
            assigned[i] = sid
            used_sids.add(sid)
            w = work[i].get("world")
            if w is not None:
                assigned_world[sid] = (float(w[0]), float(w[1]))

        def assigned_boxes() -> dict[int, tuple[int, int, int, int]]:
            return {
                assigned[j]: work[j]["xyxy"]
                for j in assigned
                if work[j].get("xyxy") is not None
            }

        # 2b) Nested / split YOLO box of someone already assigned (door edge).
        for i in range(len(work)):
            if i in assigned or i in pending_unknown:
                continue
            sid = self._overlap_sid(work[i]["xyxy"], assigned_boxes())
            if sid is None:
                continue
            assigned[i] = sid
            used_sids.add(sid)
            w = work[i].get("world")
            if w is not None:
                assigned_world[sid] = (float(w[0]), float(w[1]))

        # 3) Gallery rematch only after spatial miss (stricter identity reclaim).
        for i in range(len(work)):
            if i in assigned or i in pending_unknown:
                continue
            raw_i = work[i].get("track_id")
            raw_i = int(raw_i) if raw_i is not None else None
            probe = feat_at(i, force=True)
            self._write_temp_compare(
                frame, work[i].get("xyxy"), raw=raw_i, tag="current"
            )
            sid, _sim = self._select_rematch(
                probe, work[i].get("world"), frame_idx, used_sids
            )
            if sid is None:
                continue
            assigned[i] = sid
            used_sids.add(sid)
            w = work[i].get("world")
            if w is not None:
                assigned_world[sid] = (float(w[0]), float(w[1]))

        # 3b) If crossed IDs score higher, swap them back (high margin only).
        if (
            identity_audit_due
            and len(assigned) >= 2
            and len(set(assigned.values())) >= 2
        ):
            self._fix_id_swaps(work, assigned, from_raw, assigned_world, feat_at)
            used_sids = set(assigned.values())

        # Recovery/rematch above can introduce a crossed pair that was not
        # present during the first raw-binding pass.
        self._audit_color_bindings(
            work,
            assigned,
            used_sids,
            assigned_world,
            frame,
            drop_unknown=True,
        )

        out: list[dict] = []
        seen_raw: set[int] = set()
        for i, det in enumerate(work):
            d = dict(det)
            raw = d.get("track_id")
            if raw is not None:
                raw = int(raw)
                seen_raw.add(raw)
                self._raw_last_frame[raw] = frame_idx
                self._raw_miss[raw] = 0

            if i in pending_unknown:
                continue
            if i in assigned:
                sid = assigned[i]
            else:
                # Temp current look → must miss ALL past ID photos before mint.
                probe = feat_at(i, force=True)
                self._write_temp_compare(frame, d.get("xyxy"), raw=raw, tag="current")
                if raw is not None and probe is not None:
                    self._pending_feat[raw] = self._blend_feat(
                        self._pending_feat.get(raw), probe, alpha=0.7
                    )
                    probe = self._pending_feat[raw]
                cand_sid, _cand_sim = self._select_rematch(
                    probe, d.get("world"), frame_idx, used_sids
                )
                if cand_sid is not None:
                    sid = cand_sid
                else:
                    occ_sid = self._short_gap_recover(
                        d.get("world"),
                        frame_idx,
                        used_sids,
                        probe,
                        color_feat=color_at(i),
                    )
                    if occ_sid is not None:
                        sid = occ_sid
                    else:
                        merge_sid = self._merge_duplicate(
                            d.get("world"),
                            probe,
                            used_sids,
                            assigned_world,
                            xyxy=d.get("xyxy"),
                            assigned_boxes=assigned_boxes(),
                        )
                        if merge_sid is not None:
                            sid = merge_sid
                        else:
                            reuse_sid = self._reuse_free_sid(
                                d.get("world"),
                                frame_idx,
                                used_sids,
                                probe,
                                color_feat=color_at(i),
                            )
                            if reuse_sid is not None:
                                sid = reuse_sid
                            else:
                                if raw is None:
                                    # Untracked box — never mint.
                                    continue
                                hits = self._raw_hits.get(raw, 0) + 1
                                self._raw_hits[raw] = hits
                                if hits < self.min_hits:
                                    continue
                                final_sid, _final_sim = self._select_rematch(
                                    probe,
                                    d.get("world"),
                                    frame_idx,
                                    used_sids,
                                )
                                if final_sid is None:
                                    final_sid = self._short_gap_recover(
                                        d.get("world"),
                                        frame_idx,
                                        used_sids,
                                        probe,
                                        color_feat=color_at(i),
                                    )
                                if final_sid is None:
                                    final_sid = self._merge_duplicate(
                                        d.get("world"),
                                        probe,
                                        used_sids,
                                        assigned_world,
                                        xyxy=d.get("xyxy"),
                                        assigned_boxes=assigned_boxes(),
                                    )
                                if final_sid is None:
                                    final_sid = self._reuse_free_sid(
                                        d.get("world"),
                                        frame_idx,
                                        used_sids,
                                        probe,
                                        color_feat=color_at(i),
                                    )
                                if final_sid is not None:
                                    sid = final_sid
                                elif self._should_refuse_mint(frame_idx, len(work)):
                                    continue
                                else:
                                    sid = self._alloc_stable_id()
                                    recent_s = int(self.fps * 6)
                                    had_recent = any(
                                        frame_idx - int(m["frame"]) <= recent_s
                                        for m in self._stable.values()
                                    )
                                    if not had_recent:
                                        self._freeze_other_galleries(sid, feat_at(i))

            if raw is not None:
                self._raw_to_stable[raw] = sid
                self._raw_hits[raw] = max(self._raw_hits.get(raw, 0), self.min_hits)
                self._pending_feat.pop(raw, None)
            # Hard rule: two people in the same frame never share one ID —
            # unless the extra box is a split of the same body.
            if sid in used_sids and i not in assigned:
                boxes_now = assigned_boxes()
                overlap = (
                    self._overlap_sid(d["xyxy"], boxes_now)
                    if d.get("xyxy") is not None
                    else None
                )
                if overlap is not None:
                    if raw is not None:
                        self._raw_to_stable[raw] = overlap
                    continue
                probe = feat_at(i, force=True)
                alt, _ = self._select_rematch(
                    probe, d.get("world"), frame_idx, used_sids
                )
                if alt is None:
                    alt = self._reuse_free_sid(
                        d.get("world"),
                        frame_idx,
                        used_sids,
                        probe,
                        color_feat=color_at(i),
                    )
                if alt is None:
                    if self._should_refuse_mint(frame_idx, len(work)):
                        continue
                    alt = self._alloc_stable_id()
                sid = alt
                if raw is not None:
                    self._raw_to_stable[raw] = sid
            wx, wy = d.get("world", (0.0, 0.0))
            is_first_for_sid = sid not in self._stable
            if is_first_for_sid:
                self._sid_born_frame[sid] = frame_idx
            prev = dict(self._stable.get(sid, {}))
            xyxy = d.get("xyxy")
            other_boxes = [
                tuple(int(v) for v in work[j]["xyxy"])
                for j in range(len(work))
                if j != i and work[j].get("xyxy") is not None
            ]
            crop_clean = bool(xyxy) and not self._gallery_conflicts_others(
                xyxy, other_boxes
            )
            # Multi-person appearance is already audited once per second.
            # Track the last real embedding frame; modulo arithmetic misses
            # forever when stride=5 yields frames 1,6,11,...
            embed_every = int(self.fps) if len(work) >= 2 else 10
            last_embed = self._sid_last_embed_frame.get(sid, -10**9)
            need_feat = (
                is_first_for_sid
                or i not in from_raw
                or frame_idx - last_embed >= embed_every
            )
            cur_feat = feat_at(i, force=need_feat) if need_feat else None
            if cur_feat is not None:
                self._sid_last_embed_frame[sid] = frame_idx
            # If this crop looks like another ID, do not overwrite classmate gallery.
            can_update_gallery = self._gallery_update_allowed(sid, cur_feat) if (
                cur_feat is not None and not is_first_for_sid
            ) else bool(cur_feat is not None)
            # ByteTrack stuck classmate ID on me → switch display ID when free.
            if (
                cur_feat is not None
                and not is_first_for_sid
                and not can_update_gallery
            ):
                better = self._preferred_sid_for_feat(
                    cur_feat, sid, used_sids - {sid}
                )
                if better is not None and better not in used_sids:
                    used_sids.discard(sid)
                    sid = better
                    is_first_for_sid = sid not in self._stable
                    if is_first_for_sid:
                        self._sid_born_frame[sid] = frame_idx
                    prev = dict(self._stable.get(sid, {}))
                    can_update_gallery = (
                        True
                        if is_first_for_sid
                        else self._gallery_update_allowed(sid, cur_feat)
                    )
                    if raw is not None:
                        self._raw_to_stable[raw] = sid
            # Prefer switching to a free better ID; never collide with used_sids.
            if sid in used_sids and i not in assigned:
                boxes_now = assigned_boxes()
                overlap = (
                    self._overlap_sid(d["xyxy"], boxes_now)
                    if d.get("xyxy") is not None
                    else None
                )
                if overlap is not None:
                    if raw is not None:
                        self._raw_to_stable[raw] = overlap
                    continue
                alt = self._reuse_free_sid(
                    d.get("world"),
                    frame_idx,
                    used_sids,
                    cur_feat,
                    color_feat=color_at(i),
                )
                if alt is None:
                    if self._should_refuse_mint(frame_idx, len(work)):
                        continue
                    alt = self._alloc_stable_id()
                sid = alt
                is_first_for_sid = sid not in self._stable
                if is_first_for_sid:
                    self._sid_born_frame[sid] = frame_idx
                prev = dict(self._stable.get(sid, {}))
                can_update_gallery = cur_feat is not None
                if raw is not None:
                    self._raw_to_stable[raw] = sid
            allow_new_proto = (
                (i in from_raw or is_first_for_sid)
                and not prev.get("outfit_frozen", False)
                and can_update_gallery
                and crop_clean
            )
            # Never enroll a look that barely matches this ID (swap pollution).
            if (
                allow_new_proto
                and cur_feat is not None
                and not is_first_for_sid
                and self._proto_list(prev)
            ):
                if self._best_proto_sim(cur_feat, prev) < self.soft_appear_thresh:
                    allow_new_proto = False
            if cur_feat is not None and can_update_gallery and crop_clean:
                prev, snap_i, is_new_proto = self._update_prototypes(
                    prev, cur_feat, allow_new=allow_new_proto
                )
            else:
                snap_i, is_new_proto = None, False
            prev["frame"] = frame_idx
            prev["wx"] = float(wx)
            prev["wy"] = float(wy)
            self._stable[sid] = prev
            self._sid_exit_border[sid] = self._box_at_border(d, frame)
            used_sids.add(sid)
            assigned_world[sid] = (float(wx), float(wy))
            conf = d.get("conf")
            conf_f = float(conf) if conf is not None else None
            # Save gallery only from a clean, non-overlapping person box.
            gallery_feat = cur_feat
            needs_first_gallery = crop_clean and (
                is_first_for_sid or sid not in self._gallery_first_saved
            )
            if needs_first_gallery and gallery_feat is None:
                gallery_feat = feat_at(i, force=True)
                if gallery_feat is not None:
                    self._sid_last_embed_frame[sid] = frame_idx
            # Keep retrying first.jpg until a usable shot is saved.
            if needs_first_gallery:
                self._update_gallery_best(
                    sid,
                    frame,
                    xyxy,
                    conf_f,
                    frame_idx,
                    capture_first=True,
                    others=other_boxes,
                    feat=gallery_feat,
                )
            elif gallery_feat is not None and crop_clean and can_update_gallery and (
                (is_new_proto and snap_i)
                or self._matches_enrollment(sid, gallery_feat)
            ):
                proto_i = snap_i if is_new_proto else None
                self._update_gallery_best(
                    sid,
                    frame,
                    xyxy,
                    conf_f,
                    frame_idx,
                    also_proto=proto_i,
                    others=other_boxes,
                    feat=gallery_feat,
                )
            d["raw_track_id"] = raw
            d["track_id"] = sid
            self._sid_last_real[int(sid)] = frame_idx
            out.append(d)

        out = self._collapse_split_outputs(out, frame_idx)
        self._dump_review(out, frame, frame_idx)
        out = self._hold_missing_interior(out, frame_idx, frame)

        for raw in list(self._raw_hits.keys()):
            if raw in seen_raw or raw in self._raw_to_stable:
                continue
            miss = self._raw_miss.get(raw, 0) + 1
            self._raw_miss[raw] = miss
            if miss >= 3:
                self._raw_hits[raw] = 0
                self._pending_feat.pop(raw, None)

        if out:
            self._last_out = [dict(d) for d in out]
            self._last_out_frame = frame_idx
            return out
        return self._coasted_empty_out(frame_idx, frame)

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

import shutil
from pathlib import Path

import cv2
import numpy as np


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
        self._raw_to_stable: dict[int, int] = {}
        self._stable: dict[int, dict] = {}
        self._raw_hits: dict[int, int] = {}
        self._raw_last_frame: dict[int, int] = {}
        self._raw_miss: dict[int, int] = {}
        self._pending_feat: dict[int, np.ndarray] = {}  # raw -> EMA feat (temp)
        self._next_id = 1
        self._last_out: list[dict] = []
        self._last_out_frame = 0
        self._gallery_latest_at: dict[int, int] = {}
        self._gallery_first_saved: set[int] = set()
        self._gallery_best_area: dict[int, int] = {}
        self._gallery_best: dict[int, tuple[float, np.ndarray]] = {}
        self._gallery_first_wh: dict[int, tuple[int, int]] = {}
        self._gallery_first_feat: dict[int, np.ndarray] = {}
        self._sid_born_frame: dict[int, int] = {}
        self._unbind_votes: dict[int, tuple[str, int]] = {}
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
    ) -> int | None:
        """Collapse a second YOLO box of the *same* person only.

        Must NOT merge two nearby people (adjacent desks). Require nearly the
        same foot point; appearance alone is never enough.
        """
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
            if dist > 55.0:
                continue
            if dist <= 35.0:
                score = dist
            else:
                # 35–55cm: only if appearance also strongly agrees.
                if feat is None or meta is None:
                    continue
                sim = self._best_proto_sim(feat, meta)
                if sim < self.appear_thresh + 0.12:
                    continue
                score = dist
            if best is None or score < best[0]:
                best = (score, sid)
        return None if best is None else best[1]

    def _short_gap_recover(
        self,
        world: tuple[float, float] | None,
        frame_idx: int,
        used_sids: set[int],
        feat: np.ndarray | None = None,
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
            if feat is not None:
                sim = self._best_proto_sim(feat, meta)
                # Only reject if clearly another person after >1.5s.
                if sim < 0.10 and gap > int(self.fps * 1.5):
                    continue
            cand = (gap, dist, sid)
            if best is None or cand < best:
                best = cand
        return None if best is None else best[2]

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
        parsed = self._box_wh(frame, xyxy)
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
    def _gallery_conflicts_others(
        cls,
        xyxy: tuple[int, int, int, int],
        others: list[tuple[int, int, int, int]] | None,
    ) -> bool:
        """True if this box already swallowed (or heavily overlaps) another person."""
        if not others:
            return False
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        area = max(1, (x2 - x1) * (y2 - y1))
        for ob in others:
            ox1, oy1, ox2, oy2 = [int(v) for v in ob]
            # Any meaningful IoU → two people too close for a clean gallery shot.
            if cls._box_iou(xyxy, ob) >= 0.05:
                return True
            # Other person's center inside my box (classic mega-box contamination).
            cx = 0.5 * (ox1 + ox2)
            cy = 0.5 * (oy1 + oy2)
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return True
            # My box covers a large fraction of the other person.
            ix1, iy1 = max(x1, ox1), max(y1, oy1)
            ix2, iy2 = min(x2, ox2), min(y2, oy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            oarea = max(1, (ox2 - ox1) * (oy2 - oy1))
            if inter / float(oarea) >= 0.35 or inter / float(area) >= 0.20:
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
        if self._crop_has_split_outfits(crop):
            return False
        score = self._gallery_person_score(frame, xyxy, conf, others=others)

        if capture_first and sid not in self._gallery_first_saved:
            # Wait for a usable enrollment shot (not blurry / not multi-person).
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if sharp < 40.0 or score <= 0.0:
                return False
            if self._crop_has_split_outfits(crop):
                return False
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
            return True

        if sid not in self._gallery_first_saved:
            return False
        if not self._crop_looks_like_one_person(
            sid, crop, frame, xyxy, conf, others=others
        ):
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

        idxs = list(assigned.keys())
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
        # score, sim, -dist, sid
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
            cand = (score, sim, -dist, sid)
            if best is None or cand > best:
                best = cand
        if best is None:
            return None, -1.0
        return best[3], best[1]

    def apply(
        self,
        dets: list[dict],
        frame_idx: int,
        frame: np.ndarray | None = None,
    ) -> list[dict]:
        self._expire(frame_idx)
        if not dets:
            if (
                self._last_out
                and self.coast_frames > 0
                and frame_idx - self._last_out_frame <= self.coast_frames
            ):
                return [dict(d) for d in self._last_out]
            return []

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

        assigned: dict[int, int] = {}
        used_sids: set[int] = set()
        from_raw: set[int] = set()  # continuous ByteTrack → may learn new looks
        assigned_world: dict[int, tuple[float, float]] = {}

        # 1) Continuous YOLO/ByteTrack box → keep the same stable ID
        #    (clothes change later records a photo, does not mint).
        for i, det in enumerate(work):
            raw = det.get("track_id")
            if raw is None:
                continue
            raw = int(raw)
            sid = self._raw_to_stable.get(raw)
            if sid is None or sid not in self._stable or sid in used_sids:
                continue
            assigned[i] = sid
            used_sids.add(sid)
            from_raw.add(i)
            w = det.get("world")
            if w is not None:
                assigned_world[sid] = (float(w[0]), float(w[1]))

        # 1b) Catch ByteTrack body swaps early (two light tops look similar).
        if len(work) >= 2:
            self._audit_raw_bindings(
                work,
                assigned,
                used_sids,
                from_raw,
                assigned_world,
                feat_at,
                frame_idx,
            )

        # 2) Short-gap spatial recovery FIRST (stop ID thrash on brief drops).
        for i in range(len(work)):
            if i in assigned:
                continue
            sid = self._short_gap_recover(
                work[i].get("world"), frame_idx, used_sids, feat_at(i)
            )
            if sid is None:
                continue
            assigned[i] = sid
            used_sids.add(sid)
            w = work[i].get("world")
            if w is not None:
                assigned_world[sid] = (float(w[0]), float(w[1]))

        # 3) Gallery rematch only after spatial miss (stricter identity reclaim).
        for i in range(len(work)):
            if i in assigned:
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
        if len(assigned) >= 2:
            self._fix_id_swaps(work, assigned, from_raw, assigned_world, feat_at)
            used_sids = set(assigned.values())

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
                        d.get("world"), frame_idx, used_sids, probe
                    )
                    if occ_sid is not None:
                        sid = occ_sid
                    else:
                        merge_sid = self._merge_duplicate(
                            d.get("world"), probe, used_sids, assigned_world
                        )
                        if merge_sid is not None:
                            sid = merge_sid
                        else:
                            if raw is None:
                                hits = self.min_hits
                            else:
                                hits = self._raw_hits.get(raw, 0) + 1
                                self._raw_hits[raw] = hits
                            if hits < self.min_hits:
                                # Still collecting temp — do not mint yet.
                                continue
                            # Final gallery check (no soft last_chance — that minted ID3/ID4).
                            final_sid, _final_sim = self._select_rematch(
                                probe, d.get("world"), frame_idx, used_sids
                            )
                            if final_sid is None:
                                final_sid = self._short_gap_recover(
                                    d.get("world"), frame_idx, used_sids, probe
                                )
                            if final_sid is None:
                                final_sid = self._merge_duplicate(
                                    d.get("world"), probe, used_sids, assigned_world
                                )
                            if final_sid is not None:
                                sid = final_sid
                            else:
                                # No past ID photo matched → mint new ID.
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
            # Hard rule: two people in the same frame never share one ID.
            if sid in used_sids and i not in assigned:
                probe = feat_at(i, force=True)
                alt, _ = self._select_rematch(
                    probe, d.get("world"), frame_idx, used_sids
                )
                sid = alt if alt is not None else self._alloc_stable_id()
                if raw is not None:
                    self._raw_to_stable[raw] = sid
            wx, wy = d.get("world", (0.0, 0.0))
            is_first_for_sid = sid not in self._stable
            if is_first_for_sid:
                self._sid_born_frame[sid] = frame_idx
            prev = dict(self._stable.get(sid, {}))
            # Embed less often with 2+ people (OSNet CPU bottleneck).
            embed_every = 15 if len(work) >= 2 else 10
            need_feat = (
                is_first_for_sid
                or i not in from_raw
                or (frame_idx % embed_every) == 0
            )
            cur_feat = feat_at(i, force=need_feat) if need_feat else None
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
                sid = self._alloc_stable_id()
                is_first_for_sid = True
                self._sid_born_frame[sid] = frame_idx
                prev = {}
                can_update_gallery = cur_feat is not None
                if raw is not None:
                    self._raw_to_stable[raw] = sid
            allow_new_proto = (
                (i in from_raw or is_first_for_sid)
                and not prev.get("outfit_frozen", False)
                and can_update_gallery
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
            if cur_feat is not None and can_update_gallery:
                prev, snap_i, is_new_proto = self._update_prototypes(
                    prev, cur_feat, allow_new=allow_new_proto
                )
            else:
                snap_i, is_new_proto = None, False
            prev["frame"] = frame_idx
            prev["wx"] = float(wx)
            prev["wy"] = float(wy)
            self._stable[sid] = prev
            used_sids.add(sid)
            assigned_world[sid] = (float(wx), float(wy))
            xyxy = d.get("xyxy")
            conf = d.get("conf")
            conf_f = float(conf) if conf is not None else None
            other_boxes = [
                tuple(int(v) for v in work[j]["xyxy"])
                for j in range(len(work))
                if j != i and work[j].get("xyxy") is not None
            ]
            # Hard rule: never dump while YOLO sees 2+ people in this frame.
            alone_in_frame = len(work) < 2 and not other_boxes
            # Near-camera mega-box while classmate was just on screen → usually
            # still has their limb/shoulder in the crop; skip gallery write.
            mega_box = False
            if xyxy is not None and frame is not None:
                parsed = self._box_wh(frame, xyxy)
                if parsed is not None:
                    _x1, _y1, _x2, _y2, bw, bh = parsed
                    fh, fw = frame.shape[:2]
                    mega_box = bw > int(0.11 * fw) or (bw * bh) > int(
                        0.07 * fw * fh
                    )
            alone_clean = alone_in_frame and not (
                mega_box and self._other_id_recent(sid, frame_idx)
            )
            # latest.jpg also waits until the other ID has walked away, and
            # must still match enrollment look (blocks ID1←white-shirt overwrite).
            clear_of_neighbors = alone_clean and not self._nearby_other_person(
                sid, d.get("world"), frame_idx
            )
            gallery_feat = cur_feat if cur_feat is not None else feat_at(i, force=True)
            # Keep retrying first.jpg until a clean single-person shot is saved.
            if alone_clean and (
                is_first_for_sid or sid not in self._gallery_first_saved
            ):
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
            elif (
                clear_of_neighbors
                and can_update_gallery
                and (
                    (is_new_proto and snap_i)
                    or self._matches_enrollment(sid, gallery_feat)
                )
                and self._gallery_person_score(
                    frame, xyxy, conf_f, others=other_boxes
                )
                > 0.0
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
            out.append(d)

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
        elif (
            self._last_out
            and self.coast_frames > 0
            and frame_idx - self._last_out_frame <= self.coast_frames
        ):
            return [dict(d) for d in self._last_out]
        return out

"""Stable person ID remapping: appearance gallery first (leave-time independent)."""

from __future__ import annotations

import cv2
import numpy as np


class StableIdMapper:
    """Remap ByteTrack IDs using a session appearance gallery.

    Leave duration is unrestricted within one run. Reappearing people are
    rebound by Re-ID/HSV similarity. A **new** ID is issued only after several
    confirmed hits still fail the gallery (avoids one bad crop creating ID2/ID3
    that later pollutes matching). Among several gallery hits, the **oldest**
    ID wins so duplicates collapse toward the first identity.
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
        coast_frames: int = 60,
        sticky_frames: int = 45,
        soft_appear_thresh: float | None = None,
    ) -> None:
        self.max_dist_cm = max_dist_cm
        self.max_gap_frames = max_gap_frames
        self.max_speed_cm_s = max_speed_cm_s
        self.appear_thresh = appear_thresh
        # Below hard thresh but still "probably same" → wait, do not mint new ID.
        self.soft_appear_thresh = (
            appear_thresh * 0.72 if soft_appear_thresh is None else soft_appear_thresh
        )
        self.fps = max(1.0, fps)
        self.single_person = single_person
        self.min_hits = max(1, int(min_hits))
        self.encoder = encoder
        self.coast_frames = max(0, int(coast_frames))
        self.sticky_frames = max(0, int(sticky_frames))
        self._raw_to_stable: dict[int, int] = {}
        self._stable: dict[int, dict] = {}
        self._raw_hits: dict[int, int] = {}
        self._raw_last_frame: dict[int, int] = {}
        self._raw_miss: dict[int, int] = {}
        self._pending_feat: dict[int, np.ndarray] = {}  # raw -> EMA feat while waiting
        self._next_id = 1
        self._last_out: list[dict] = []
        self._last_out_frame = 0

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
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = xyxy
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
        if x2 - x1 < 8 or y2 - y1 < 16:
            return None
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

    def _reach_limit_cm(self, gap_frames: int) -> float:
        gap_s = max(0, gap_frames) / self.fps
        return min(self.max_dist_cm, self.max_speed_cm_s * gap_s + 80.0)

    def _gallery_hits(
        self, feat: np.ndarray | None, used_sids: set[int], thresh: float
    ) -> list[tuple[float, int]]:
        """Return [(sim, sid), ...] with sim >= thresh, oldest sid preferred later."""
        hits: list[tuple[float, int]] = []
        if feat is None:
            return hits
        for sid, meta in self._stable.items():
            if sid in used_sids:
                continue
            sim = self._appear_sim(feat, meta.get("feat"))
            if sim >= thresh:
                hits.append((sim, sid))
        # Highest sim first; tie → older (smaller) sid.
        hits.sort(key=lambda t: (-t[0], t[1]))
        return hits

    def _pick_gallery_id(
        self, feat: np.ndarray | None, used_sids: set[int], thresh: float
    ) -> tuple[int | None, float]:
        hits = self._gallery_hits(feat, used_sids, thresh)
        if not hits:
            return None, -1.0
        # Among near-best sims, prefer oldest ID to collapse duplicates.
        best_sim = hits[0][0]
        near = [sid for sim, sid in hits if sim >= best_sim - 0.03]
        sid = min(near)
        sim = max(s for s, i in hits if i == sid)
        return sid, sim

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

        feats: list[np.ndarray | None] = []
        for det in work:
            if frame is None:
                feats.append(None)
            else:
                feats.append(self.appearance_feat(frame, det["xyxy"], encoder=self.encoder))

        assigned: dict[int, int] = {}
        used_sids: set[int] = set()

        # 1) Raw ByteTrack continuity, but allow rebind to an *older* gallery ID
        #    if appearance strongly prefers it (collapse ID2→ID1 duplicates).
        for i, det in enumerate(work):
            raw = det.get("track_id")
            if raw is None:
                continue
            raw = int(raw)
            sid = self._raw_to_stable.get(raw)
            if sid is None or sid not in self._stable or sid in used_sids:
                continue
            gal_sid, gal_sim = self._pick_gallery_id(
                feats[i], used_sids - {sid}, self.appear_thresh
            )
            cur_sim = self._appear_sim(feats[i], self._stable[sid].get("feat"))
            if (
                gal_sid is not None
                and gal_sid < sid
                and gal_sim >= self.appear_thresh
                and gal_sim >= cur_sim - 0.02
            ):
                assigned[i] = gal_sid
                used_sids.add(gal_sid)
                self._raw_to_stable[raw] = gal_sid
            else:
                assigned[i] = sid
                used_sids.add(sid)

        # 2) Appearance gallery (any leave duration) — primary rule.
        for i in range(len(work)):
            if i in assigned:
                continue
            sid, sim = self._pick_gallery_id(feats[i], used_sids, self.appear_thresh)
            if sid is not None:
                assigned[i] = sid
                used_sids.add(sid)

        # 3) Short-gap motion fallback only.
        for i in range(len(work)):
            if i in assigned:
                continue
            world = work[i].get("world")
            if world is None:
                continue
            best = None
            for sid, meta in self._stable.items():
                if sid in used_sids:
                    continue
                gap = frame_idx - int(meta["frame"])
                if gap > max(self.sticky_frames, 45):
                    continue
                dist = self._dist(world, (float(meta["wx"]), float(meta["wy"])))
                limit = self._reach_limit_cm(gap)
                if dist <= max(limit, 200.0):
                    cand = (dist, sid)
                    if best is None or cand < best:
                        best = cand
            if best is not None:
                assigned[i] = best[1]
                used_sids.add(best[1])

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
                self._pending_feat[raw] = self._blend_feat(
                    self._pending_feat.get(raw), feats[i], alpha=0.7
                )

            if i in assigned:
                sid = assigned[i]
            else:
                # Soft gallery: probably same person → bind, never mint new yet.
                probe = feats[i]
                if raw is not None and raw in self._pending_feat:
                    probe = self._pending_feat[raw]
                soft_sid, soft_sim = self._pick_gallery_id(
                    probe, used_sids, self.soft_appear_thresh
                )
                if soft_sid is not None:
                    sid = soft_sid
                else:
                    # Need several misses against gallery before a brand-new ID.
                    if raw is None:
                        hits = self.min_hits
                    else:
                        hits = self._raw_hits.get(raw, 0) + 1
                        self._raw_hits[raw] = hits
                    if hits < self.min_hits:
                        # Not confirmed as new — hide this frame (coast may cover).
                        continue
                    # Final gallery check with accumulated pending feat.
                    final_sid, final_sim = self._pick_gallery_id(
                        probe, used_sids, self.soft_appear_thresh
                    )
                    if final_sid is not None:
                        sid = final_sid
                    else:
                        sid = self._alloc_stable_id()

            if raw is not None:
                self._raw_to_stable[raw] = sid
                self._raw_hits[raw] = max(self._raw_hits.get(raw, 0), self.min_hits)
                self._pending_feat.pop(raw, None)
            wx, wy = d.get("world", (0.0, 0.0))
            prev = self._stable.get(sid, {})
            feat = self._blend_feat(prev.get("feat"), feats[i], alpha=0.85)
            self._stable[sid] = {
                "frame": frame_idx,
                "wx": float(wx),
                "wy": float(wy),
                "feat": feat,
            }
            used_sids.add(sid)
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

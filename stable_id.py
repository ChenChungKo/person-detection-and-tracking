"""Stable person ID remapping: appearance gallery first (leave-time independent)."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np


class StableIdMapper:
    """Remap ByteTrack IDs using a session appearance gallery.

    Leave duration is unrestricted within one run. Reappearing people are
    rebound by Re-ID/HSV similarity. Each person keeps **multiple appearance
    prototypes** (unlimited by default; e.g. with / without jacket) so clothing
    change during a continuous track still rematches after leave/re-enter.
    The first photo for a new ID is saved immediately as ``first.jpg``.

    A **new** ID is issued only after several confirmed hits still fail the
    gallery. Among several near-equal gallery hits, the **most recently seen**
    ID wins (avoids flipping ID3→ID1 after a clothing-change enrollment).
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
        self._pending_feat: dict[int, np.ndarray] = {}  # raw -> EMA feat while waiting
        self._next_id = 1
        self._last_out: list[dict] = []
        self._last_out_frame = 0
        self._gallery_latest_at: dict[int, int] = {}
        self._gallery_first_saved: set[int] = set()
        if self.gallery_dir is not None:
            if self.gallery_dir.exists():
                shutil.rmtree(self.gallery_dir)
            self.gallery_dir.mkdir(parents=True, exist_ok=True)

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
    def _crop_person(
        frame: np.ndarray, xyxy: tuple[int, int, int, int]
    ) -> np.ndarray | None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 - x1 < 8 or y2 - y1 < 16:
            return None
        return frame[y1:y2, x1:x2].copy()

    def _save_gallery_image(
        self,
        sid: int,
        name: str,
        frame: np.ndarray | None,
        xyxy: tuple[int, int, int, int] | None,
    ) -> None:
        if self.gallery_dir is None or frame is None or xyxy is None:
            return
        crop = self._crop_person(frame, xyxy)
        if crop is None:
            return
        folder = self.gallery_dir / f"ID{sid:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        cv2.imwrite(str(path), crop)

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
            limit = max(self._reach_limit_cm(gap), 150.0)
            if dist > limit:
                return False
        return True

    def _select_rematch(
        self,
        feat: np.ndarray | None,
        world: tuple[float, float] | None,
        frame_idx: int,
        used_sids: set[int],
        last_chance: bool = False,
    ) -> tuple[int | None, float]:
        """Best gallery rematch: highest sim, then most recently seen."""
        recent_shown = {
            int(d["track_id"])
            for d in self._last_out
            if d.get("track_id") is not None
        }
        best: tuple[float, float, int, int] | None = None
        # score, sim, last_frame, sid
        for sid, meta in self._stable.items():
            if sid in used_sids:
                continue
            if not self._rematch_allowed(
                sid, feat, world, frame_idx, last_chance=last_chance
            ):
                continue
            sim = self._best_proto_sim(feat, meta)
            last_f = int(meta["frame"])
            score = sim + (0.02 if sid in recent_shown else 0.0)
            cand = (score, sim, last_f, sid)
            if best is None or cand[:3] > best[:3]:
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

        feats: list[np.ndarray | None] = []
        for det in work:
            if frame is None:
                feats.append(None)
            else:
                feats.append(self.appearance_feat(frame, det["xyxy"], encoder=self.encoder))

        assigned: dict[int, int] = {}
        used_sids: set[int] = set()
        from_raw: set[int] = set()  # continuous ByteTrack → may learn new looks

        # 1) Raw ByteTrack continuity — keep the bound stable ID.
        #    Do not collapse a newly minted outfit ID (e.g. jacket→ID2) back
        #    into an older ID that already stored a similar look.
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

        # 2) Short-gap recovery (pillar / brief occlusion): keep ID by floor
        #    proximity even when Re-ID embedding is corrupted by a partial crop.
        for i in range(len(work)):
            if i in assigned:
                continue
            sid = self._short_gap_recover(
                work[i].get("world"), frame_idx, used_sids, feats[i]
            )
            if sid is None:
                continue
            assigned[i] = sid
            used_sids.add(sid)

        # 3) Appearance gallery (longer leave / re-entry) + spatial sanity.
        for i in range(len(work)):
            if i in assigned:
                continue
            sid, _sim = self._select_rematch(
                feats[i], work[i].get("world"), frame_idx, used_sids
            )
            if sid is None:
                continue
            assigned[i] = sid
            used_sids.add(sid)

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
                probe = feats[i]
                if raw is not None and raw in self._pending_feat:
                    probe = self._pending_feat[raw]
                cand_sid, _cand_sim = self._select_rematch(
                    probe, d.get("world"), frame_idx, used_sids
                )
                if cand_sid is not None:
                    sid = cand_sid
                else:
                    # Occlusion last chance before counting toward a new ID.
                    occ_sid = self._short_gap_recover(
                        d.get("world"), frame_idx, used_sids, probe
                    )
                    if occ_sid is not None:
                        sid = occ_sid
                    else:
                        if raw is None:
                            hits = self.min_hits
                        else:
                            hits = self._raw_hits.get(raw, 0) + 1
                            self._raw_hits[raw] = hits
                        if hits < self.min_hits:
                            continue
                        final_sid, _final_sim = self._select_rematch(
                            probe, d.get("world"), frame_idx, used_sids
                        )
                        if final_sid is None:
                            final_sid, _final_sim = self._select_rematch(
                                probe,
                                d.get("world"),
                                frame_idx,
                                used_sids,
                                last_chance=True,
                            )
                        if final_sid is None:
                            final_sid = self._short_gap_recover(
                                d.get("world"), frame_idx, used_sids, probe
                            )
                        if final_sid is not None:
                            sid = final_sid
                        else:
                            sid = self._alloc_stable_id()
                            # Outfit split only if nobody was around recently
                            # (not a pillar flicker that just lost ID1).
                            recent_s = int(self.fps * 6)
                            had_recent = any(
                                frame_idx - int(m["frame"]) <= recent_s
                                for m in self._stable.values()
                            )
                            if not had_recent:
                                self._freeze_other_galleries(sid, feats[i])

            if raw is not None:
                self._raw_to_stable[raw] = sid
                self._raw_hits[raw] = max(self._raw_hits.get(raw, 0), self.min_hits)
                self._pending_feat.pop(raw, None)
            wx, wy = d.get("world", (0.0, 0.0))
            is_first_for_sid = sid not in self._stable
            prev = dict(self._stable.get(sid, {}))
            # Only continuous tracks may enroll a brand-new look under an ID.
            # Frozen galleries (outfit split off to a newer ID) stay closed.
            allow_new_proto = (
                (i in from_raw or is_first_for_sid)
                and not prev.get("outfit_frozen", False)
            )
            prev, snap_i, is_new_proto = self._update_prototypes(
                prev, feats[i], allow_new=allow_new_proto
            )
            prev["frame"] = frame_idx
            prev["wx"] = float(wx)
            prev["wy"] = float(wy)
            self._stable[sid] = prev
            used_sids.add(sid)
            xyxy = d.get("xyxy")
            # First time this stable ID is created: always snapshot one photo.
            if is_first_for_sid and sid not in self._gallery_first_saved:
                self._save_gallery_image(sid, "first.jpg", frame, xyxy)
                self._save_gallery_image(sid, "proto_0.jpg", frame, xyxy)
                self._save_gallery_image(sid, "latest.jpg", frame, xyxy)
                self._gallery_first_saved.add(sid)
                self._gallery_latest_at[sid] = frame_idx
            elif is_new_proto and snap_i is not None:
                self._save_gallery_image(sid, f"proto_{snap_i}.jpg", frame, xyxy)
            last_g = self._gallery_latest_at.get(sid, -10**9)
            if frame_idx - last_g >= self.gallery_latest_every:
                self._save_gallery_image(sid, "latest.jpg", frame, xyxy)
                self._gallery_latest_at[sid] = frame_idx
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

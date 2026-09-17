"""Enrollment gate: overlapping boxes must not mint Stable-IDs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from stable_id import StableIdMapper, review_frame_due

H, W = 720, 1280
# ~120x320, aspect ~2.7, well inside the gallery-quality window.
A_OVER = (100, 180, 220, 500)
B_OVER = (140, 180, 260, 500)  # IoU ~0.50 with A_OVER
A_SEP = (80, 180, 200, 500)
B_SEP = (900, 180, 1020, 500)
BLUE = (200, 60, 20)
RED = (20, 60, 200)


def _paint(frame: np.ndarray, xyxy, bgr) -> None:
    x1, y1, x2, y2 = xyxy
    frame[y1:y2, x1:x2] = bgr


def _frame(*painted: tuple) -> np.ndarray:
    frame = np.full((H, W, 3), 28, dtype=np.uint8)
    for xyxy, bgr in painted:
        _paint(frame, xyxy, bgr)
    return frame


def _det(raw: int, xyxy, world, conf: float = 0.85) -> dict:
    return {
        "track_id": raw,
        "xyxy": xyxy,
        "world": world,
        "conf": conf,
    }


def _ids(out: list[dict]) -> list:
    return [d.get("track_id") for d in out]


class EnrollWhenCleanTests(unittest.TestCase):
    def test_continuous_raw_track_is_not_downgraded_by_one_bad_crop(self) -> None:
        """A transient appearance mismatch must not draw person over the same ID."""
        mapper = StableIdMapper(min_hits=1, encoder=None, gallery_dir=None)
        first = mapper.apply(
            [_det(7, A_SEP, (100.0, 80.0))], 1, _frame((A_SEP, BLUE))
        )
        self.assertEqual(_ids(first), [1])

        # Reproduce the live failure at test4 frame 1086: BoT-SORT raw ID is
        # continuous, but an occluded crop fails the appearance guard.
        mapper._appearance_forbids_reuse = lambda *args: True
        for frame_idx in range(2, 6):
            out = mapper.apply(
                [_det(7, A_SEP, (100.0, 80.0))],
                frame_idx,
                _frame((A_SEP, RED)),
            )
            self.assertEqual(len(out), 1, f"same raw track emitted twice: {out}")
            self.assertEqual(out[0].get("raw_track_id"), 7)
            self.assertEqual(out[0].get("track_id"), 1)

    def test_coast_never_duplicates_current_raw_track(self) -> None:
        mapper = StableIdMapper(min_hits=1, encoder=None, gallery_dir=None)
        mapper._last_out = [
            {
                **_det(7, A_SEP, (100.0, 80.0)),
                "raw_track_id": 7,
                "track_id": 1,
            }
        ]
        mapper._last_out_frame = 1
        mapper._sid_last_real[1] = 1
        current = [
            {
                **_det(7, A_SEP, (100.0, 80.0)),
                "raw_track_id": 7,
                "track_id": None,
            }
        ]
        out = mapper._hold_missing_interior(current, 2, _frame())
        self.assertEqual(out, current)

    def test_new_raw_cannot_steal_briefly_missing_raws_id(self) -> None:
        """test4 58.8s: raw13 must not take ID2 while raw1 misses one frame."""
        mapper = StableIdMapper(
            min_hits=1, encoder=None, gallery_dir=None, fps=20.0
        )
        first = mapper.apply(
            [_det(7, A_SEP, (100.0, 80.0))], 1, _frame((A_SEP, BLUE))
        )
        self.assertEqual(_ids(first), [1])

        intruder = mapper.apply(
            [_det(88, B_SEP, (700.0, 500.0))], 6, _frame((B_SEP, BLUE))
        )
        by_raw = {
            d.get("raw_track_id"): d.get("track_id") for d in intruder
        }
        self.assertIsNone(by_raw.get(88), f"intruder stole ID1: {intruder}")
        self.assertEqual(mapper._sid_raw_owner.get(1), 7)

        both = mapper.apply(
            [
                _det(7, A_SEP, (100.0, 80.0)),
                _det(88, B_SEP, (700.0, 500.0)),
            ],
            11,
            _frame((A_SEP, BLUE), (B_SEP, RED)),
        )
        by_raw = {d.get("raw_track_id"): d.get("track_id") for d in both}
        self.assertEqual(by_raw.get(7), 1)
        self.assertEqual(by_raw.get(88), 2)

    def test_occluder_does_not_inherit_vacant_id1(self) -> None:
        """Walker covering ID1 must stay unlabeled; ID1 returns when clear."""
        mapper = StableIdMapper(
            min_hits=1, encoder=None, gallery_dir=None, fps=20.0
        )
        desk = A_SEP
        near = (140, 180, 260, 500)  # overlaps / near ID1's desk
        mapper.apply(
            [_det(7, desk, (100.0, 80.0))], 1, _frame((desk, BLUE))
        )
        # Only the walker is visible near the desk (ID1 occluded).
        occluded = mapper.apply(
            [_det(88, near, (130.0, 80.0))],
            6,
            _frame((near, RED)),
        )
        by_raw = {d.get("raw_track_id"): d.get("track_id") for d in occluded}
        self.assertNotEqual(
            by_raw.get(88), 1, f"walker stole vacant ID1: {occluded}"
        )
        # Original person reappears alone → reclaim ID1.
        back = mapper.apply(
            [_det(7, desk, (100.0, 80.0))],
            20,
            _frame((desk, BLUE)),
        )
        by_raw = {d.get("raw_track_id"): d.get("track_id") for d in back}
        self.assertEqual(by_raw.get(7), 1, f"ID1 not restored: {back}")

    def test_continuous_raw_teleport_does_not_keep_id1(self) -> None:
        """Same BoT-SORT raw jumping onto a differently dressed walker drops ID1."""
        mapper = StableIdMapper(
            min_hits=1, encoder=None, gallery_dir=None, fps=20.0
        )
        mapper.apply(
            [_det(7, A_SEP, (100.0, 80.0))], 1, _frame((A_SEP, BLUE))
        )
        # Someone still at the old seat + wrong clothes on the jumped raw.
        out = mapper.apply(
            [
                _det(7, B_SEP, (700.0, 500.0)),
                _det(22, A_SEP, (100.0, 80.0)),
            ],
            6,
            _frame((B_SEP, RED), (A_SEP, BLUE)),
        )
        by_raw = {d.get("raw_track_id"): d.get("track_id") for d in out}
        self.assertNotEqual(
            by_raw.get(7), 1, f"teleported raw kept ID1: {out}"
        )
        self.assertEqual(by_raw.get(22), 1, f"seat lost ID1: {out}")

    def test_same_desk_raw_renumber_keeps_id1(self) -> None:
        """BoT-SORT switching raw at the same seat must not blank ID1."""
        mapper = StableIdMapper(
            min_hits=1, encoder=None, gallery_dir=None, fps=20.0
        )
        mapper.apply(
            [_det(7, A_SEP, (100.0, 80.0))], 1, _frame((A_SEP, BLUE))
        )
        out = mapper.apply(
            [_det(99, A_SEP, (105.0, 82.0))],
            6,
            _frame((A_SEP, BLUE)),
        )
        by_raw = {d.get("raw_track_id"): d.get("track_id") for d in out}
        self.assertEqual(by_raw.get(99), 1, f"same-desk renumber lost ID1: {out}")

    def test_short_gap_id_not_blanked_by_overlap_conflict(self) -> None:
        """A recovered ID must stay labelled even if the crop overlaps a neighbor."""
        mapper = StableIdMapper(
            min_hits=1, encoder=None, gallery_dir=None, fps=20.0
        )
        mapper.apply(
            [_det(7, A_SEP, (100.0, 80.0))], 1, _frame((A_SEP, BLUE))
        )
        # New raw at the same desk, slightly overlapping a second person.
        near = (140, 180, 260, 500)
        out = mapper.apply(
            [
                _det(99, A_OVER, (105.0, 80.0)),
                _det(22, near, (160.0, 80.0)),
            ],
            6,
            _frame((A_OVER, BLUE), (near, RED)),
        )
        by_raw = {d.get("raw_track_id"): d.get("track_id") for d in out}
        self.assertEqual(
            by_raw.get(99), 1, f"recovered ID1 became person: {out}"
        )

    def test_same_person_walk_keeps_id_across_jump(self) -> None:
        """A real walk with matching clothes must not flicker off ID1."""
        mapper = StableIdMapper(
            min_hits=1, encoder=None, gallery_dir=None, fps=20.0
        )
        mapper.apply(
            [_det(7, A_SEP, (100.0, 80.0))], 1, _frame((A_SEP, BLUE))
        )
        out = mapper.apply(
            [_det(7, B_SEP, (450.0, 80.0))],
            6,
            _frame((B_SEP, BLUE)),
        )
        by_raw = {d.get("raw_track_id"): d.get("track_id") for d in out}
        self.assertEqual(by_raw.get(7), 1, f"walk lost ID1: {out}")

    def test_new_raw_can_reclaim_after_owner_reservation_expires(self) -> None:
        mapper = StableIdMapper(
            min_hits=1, encoder=None, gallery_dir=None, fps=20.0
        )
        mapper.apply(
            [_det(7, A_SEP, (100.0, 80.0))], 1, _frame((A_SEP, BLUE))
        )
        out = mapper.apply(
            [_det(88, A_SEP, (100.0, 80.0))], 25, _frame((A_SEP, BLUE))
        )
        self.assertEqual(_ids(out), [1], f"replacement raw could not reclaim: {out}")
        self.assertEqual(mapper._sid_raw_owner.get(1), 88)

    def test_recent_new_raw_cannot_teleport_into_vacant_id(self) -> None:
        """test4 frame1161: ID1 cannot jump 474cm to raw14 in 1.25s."""
        mapper = StableIdMapper(
            min_hits=1, encoder=None, gallery_dir=None, fps=20.0
        )
        mapper.apply(
            [_det(7, A_SEP, (100.0, 80.0))], 1, _frame((A_SEP, BLUE))
        )
        out = mapper.apply(
            [_det(88, B_SEP, (700.0, 500.0))],
            26,
            _frame((B_SEP, BLUE)),
        )
        by_raw = {d.get("raw_track_id"): d.get("track_id") for d in out}
        self.assertIsNone(by_raw.get(88), f"new raw teleported into ID1: {out}")
        self.assertEqual(mapper._sid_raw_owner.get(1), 7)

    def test_dirty_unbound_fragment_cannot_reclaim_vacant_id(self) -> None:
        """test4 62s: an overlapping raw14 fragment must not steal ID1."""
        mapper = StableIdMapper(
            min_hits=1, encoder=None, gallery_dir=None, fps=20.0
        )
        initial = [
            _det(7, B_SEP, (700.0, 500.0)),
            _det(22, A_SEP, (100.0, 80.0)),
        ]
        mapper.apply(initial, 1, _frame((B_SEP, BLUE), (A_SEP, RED)))

        # ID1's owner is now long enough absent to permit a legitimate reclaim,
        # but this blue fragment overlaps the still-tracked red person.
        dirty = [
            _det(22, A_OVER, (100.0, 80.0)),
            _det(88, B_OVER, (700.0, 500.0)),
        ]
        out = mapper.apply(dirty, 25, _frame((A_OVER, RED), (B_OVER, BLUE)))
        by_raw = {d.get("raw_track_id"): d.get("track_id") for d in out}
        self.assertEqual(by_raw.get(22), 2)
        self.assertIsNone(by_raw.get(88), f"dirty fragment stole ID1: {out}")

    def test_overlap_is_a_gallery_conflict(self) -> None:
        self.assertTrue(
            StableIdMapper._gallery_conflicts_others(A_OVER, [B_OVER])
        )
        self.assertFalse(
            StableIdMapper._gallery_conflicts_others(A_SEP, [B_SEP])
        )

    def test_separated_boxes_are_enrollable(self) -> None:
        frame = _frame((A_SEP, BLUE), (B_SEP, RED))
        mapper = StableIdMapper(min_hits=2, encoder=None, gallery_dir=None)
        self.assertTrue(
            mapper._can_enroll_new_id(frame, A_SEP, 0.85, [B_SEP])
        )
        self.assertTrue(
            mapper._can_enroll_new_id(frame, B_SEP, 0.85, [A_SEP])
        )
        self.assertFalse(
            mapper._can_enroll_new_id(frame, A_OVER, 0.85, [B_OVER])
        )

    def test_overlapping_people_stay_unlabeled_until_separated(self) -> None:
        mapper = StableIdMapper(min_hits=2, encoder=None, gallery_dir=None)
        overlap = [
            _det(11, A_OVER, (120.0, 80.0)),
            _det(22, B_OVER, (500.0, 80.0)),
        ]
        frame_over = _frame((A_OVER, BLUE), (B_OVER, RED))
        for frame_idx in (1, 2, 3, 4):
            out = mapper.apply(overlap, frame_idx, frame_over)
            self.assertEqual(len(out), 2, f"frame {frame_idx} dropped a person")
            self.assertEqual(
                _ids(out),
                [None, None],
                f"frame {frame_idx} minted while overlapping: {_ids(out)}",
            )

        split = [
            _det(11, A_SEP, (120.0, 80.0)),
            _det(22, B_SEP, (500.0, 80.0)),
        ]
        frame_sep = _frame((A_SEP, BLUE), (B_SEP, RED))
        out = mapper.apply(split, 5, frame_sep)
        ids = [sid for sid in _ids(out) if sid is not None]
        self.assertEqual(len(out), 2)
        self.assertEqual(len(set(ids)), 2, f"expected two new IDs, got {_ids(out)}")
        self.assertEqual(sorted(ids), [1, 2])

    def test_clean_solo_person_still_mints_after_min_hits(self) -> None:
        mapper = StableIdMapper(min_hits=2, encoder=None, gallery_dir=None)
        dets = [_det(7, A_SEP, (120.0, 80.0))]
        frame = _frame((A_SEP, BLUE))
        out1 = mapper.apply(dets, 1, frame)
        self.assertEqual(_ids(out1), [None])
        out2 = mapper.apply(dets, 2, frame)
        self.assertEqual(_ids(out2), [1])

    def test_clean_neighbor_can_mint_while_other_pair_overlaps(self) -> None:
        """Per-box gate: a separated third person is not frozen with the crowd."""
        mapper = StableIdMapper(min_hits=2, encoder=None, gallery_dir=None)
        c_box = (980, 180, 1100, 500)
        green = (20, 180, 20)
        dets = [
            _det(11, A_OVER, (120.0, 80.0)),
            _det(22, B_OVER, (220.0, 80.0)),
            _det(33, c_box, (700.0, 80.0)),
        ]
        frame = _frame((A_OVER, BLUE), (B_OVER, RED), (c_box, green))
        mapper.apply(dets, 1, frame)
        out = mapper.apply(dets, 2, frame)
        by_raw = {d.get("raw_track_id"): d.get("track_id") for d in out}
        self.assertIsNone(by_raw.get(11))
        self.assertIsNone(by_raw.get(22))
        self.assertEqual(by_raw.get(33), 1)

    def test_first_jpg_only_after_clean_mint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gallery = Path(tmp) / "gallery"
            mapper = StableIdMapper(
                min_hits=2, encoder=None, gallery_dir=str(gallery)
            )
            overlap = [
                _det(11, A_OVER, (120.0, 80.0)),
                _det(22, B_OVER, (500.0, 80.0)),
            ]
            frame_over = _frame((A_OVER, BLUE), (B_OVER, RED))
            mapper.apply(overlap, 1, frame_over)
            mapper.apply(overlap, 2, frame_over)
            self.assertFalse(any(gallery.glob("ID*/first.jpg")))

            split = [
                _det(11, A_SEP, (120.0, 80.0)),
                _det(22, B_SEP, (500.0, 80.0)),
            ]
            frame_sep = _frame((A_SEP, BLUE), (B_SEP, RED))
            out = mapper.apply(split, 3, frame_sep)
            ids = sorted(sid for sid in _ids(out) if sid is not None)
            self.assertEqual(ids, [1, 2])
            for sid in ids:
                self.assertTrue(
                    (gallery / f"ID{sid:03d}" / "first.jpg").is_file(),
                    f"missing first.jpg for ID{sid}",
                )

    def test_walk_away_reclaims_id1_instead_of_minting(self) -> None:
        mapper = StableIdMapper(min_hits=2, encoder=None, gallery_dir=None)
        desk = [_det(7, A_SEP, (100.0, 80.0))]
        frame_desk = _frame((A_SEP, BLUE))
        mapper.apply(desk, 1, frame_desk)
        self.assertEqual(_ids(mapper.apply(desk, 2, frame_desk)), [1])

        walked = [_det(99, B_SEP, (520.0, 380.0))]
        frame_walk = _frame((B_SEP, BLUE))
        out = mapper.apply(walked, 3, frame_walk)
        # An impossible one-frame position jump under a new raw ID first
        # remains unbound; this protects the old owner from ID stealing.
        by_raw = {d.get("raw_track_id"): d.get("track_id") for d in out}
        self.assertIsNone(by_raw.get(99), f"new raw stole ID1: {out}")
        # After a genuinely long absence, appearance may reclaim from anywhere.
        out = mapper.apply(walked, 205, frame_walk)
        by_raw = {d.get("raw_track_id"): d.get("track_id") for d in out}
        self.assertEqual(by_raw.get(99), 1, f"walking did not reclaim ID1: {out}")

    def test_short_walk_reclaims_instead_of_minting(self) -> None:
        """Track break while walking ~3.5m in 2s must keep ID1 (test3 ID5/6)."""
        mapper = StableIdMapper(
            min_hits=1,
            encoder=None,
            gallery_dir=None,
            fps=20.0,
            max_dist_cm=800.0,
            max_speed_cm_s=200.0,
        )
        mapper.apply(
            [_det(7, A_SEP, (100.0, 80.0))], 1, _frame((A_SEP, BLUE))
        )
        out = mapper.apply(
            [_det(88, B_SEP, (450.0, 80.0))],
            40,
            _frame((B_SEP, BLUE)),
        )
        by_raw = {d.get("raw_track_id"): d.get("track_id") for d in out}
        self.assertEqual(by_raw.get(88), 1, f"walker minted a second ID: {out}")
        self.assertNotIn(2, _ids(out))

    def test_full_person_touching_edge_is_not_an_exit(self) -> None:
        mapper = StableIdMapper(min_hits=1, encoder=None, gallery_dir=None)
        edge = (W - 130, 180, W - 2, 520)
        det = {**_det(7, edge, (500.0, 80.0)), "raw_track_id": 7, "track_id": 1}
        self.assertFalse(mapper._box_at_border(det, _frame((edge, BLUE))))
        sliver = (W - 20, 200, W - 1, 280)
        slim = {**_det(8, sliver, (500.0, 80.0)), "raw_track_id": 8, "track_id": 2}
        self.assertTrue(mapper._box_at_border(slim, _frame((sliver, BLUE))))

    def test_split_outfit_crop_does_not_enroll(self) -> None:
        mapper = StableIdMapper(min_hits=1, encoder=None, gallery_dir=None)
        box = A_SEP
        frame = np.full((H, W, 3), 28, dtype=np.uint8)
        x1, y1, x2, y2 = box
        mid = (x1 + x2) // 2
        frame[y1:y2, x1:mid] = (20, 40, 210)
        frame[y1:y2, mid:x2] = (210, 40, 20)
        other = (x1 + 40, y1 + 10, x2 + 70, y2)
        self.assertFalse(mapper._can_enroll_new_id(frame, box, 0.85, [other]))
        # Furniture next to a seated person is not a second body.
        self.assertTrue(mapper._can_enroll_new_id(frame, box, 0.85, []))


    def test_different_clothes_do_not_steal_vacant_id1(self) -> None:
        mapper = StableIdMapper(min_hits=2, encoder=None, gallery_dir=None)
        desk = [_det(7, A_SEP, (100.0, 80.0))]
        frame_desk = _frame((A_SEP, BLUE))
        mapper.apply(desk, 1, frame_desk)
        self.assertEqual(_ids(mapper.apply(desk, 2, frame_desk)), [1])

        other = [_det(99, B_SEP, (520.0, 380.0))]
        frame_other = _frame((B_SEP, RED))
        mapper.apply(other, 3, frame_other)
        out = mapper.apply(other, 4, frame_other)
        by_raw = {d.get("raw_track_id"): d.get("track_id") for d in out}
        self.assertNotEqual(by_raw.get(99), 1, f"other person stole ID1: {out}")

    def test_different_clothes_at_same_desk_do_not_inherit_id1(self) -> None:
        mapper = StableIdMapper(min_hits=2, encoder=None, gallery_dir=None)
        desk = [_det(7, A_SEP, (100.0, 80.0))]
        frame_desk = _frame((A_SEP, BLUE))
        mapper.apply(desk, 1, frame_desk)
        mapper.apply(desk, 2, frame_desk)

        other = [_det(99, A_SEP, (105.0, 85.0))]
        frame_other = _frame((A_SEP, RED))
        mapper.apply(other, 3, frame_other)
        out = mapper.apply(other, 4, frame_other)
        by_raw = {d.get("raw_track_id"): d.get("track_id") for d in out}
        self.assertNotEqual(by_raw.get(99), 1, f"desk successor stole ID1: {out}")

    def test_color_reclaim_keeps_older_same_clothes_id(self) -> None:
        mapper = StableIdMapper(min_hits=1, encoder=None, gallery_dir=None)
        dets = [
            _det(1, A_SEP, (100.0, 80.0)),
            _det(2, B_SEP, (500.0, 80.0)),
        ]
        frame = _frame((A_SEP, BLUE), (B_SEP, RED))
        mapper.apply(dets, 1, frame)
        out = mapper.apply(dets, 2, frame)
        self.assertEqual(sorted(sid for sid in _ids(out) if sid), [1, 2])
        color = mapper._color_feat_from_crop(mapper._crop_person(frame, B_SEP))
        self.assertEqual(mapper._color_reclaim(color, set()), 2)
        self.assertIsNone(mapper._color_reclaim(color, {2}))

    def test_second_box_same_clothes_merges_to_id1(self) -> None:
        mapper = StableIdMapper(min_hits=1, encoder=None, gallery_dir=None)
        desk = [_det(7, A_SEP, (100.0, 80.0))]
        frame = _frame((A_SEP, BLUE))
        mapper.apply(desk, 1, frame)
        extra = (110, 190, 210, 510)
        both = [
            _det(7, A_SEP, (100.0, 80.0)),
            _det(88, extra, (110.0, 85.0)),
        ]
        frame2 = _frame((A_SEP, BLUE), (extra, BLUE))
        out = mapper.apply(both, 2, frame2)
        ids = [sid for sid in _ids(out) if sid is not None]
        self.assertTrue(ids, f"lost both people: {_ids(out)}")
        self.assertNotIn(2, ids, f"split a second ID: {_ids(out)}")
        self.assertEqual(set(ids), {1})

    def test_adjacent_seated_classmates_get_two_ids(self) -> None:
        """A person sitting beside ID1 must mint, not stay a flickering person."""
        mapper = StableIdMapper(min_hits=1, encoder=None, gallery_dir=None)
        neighbor = (240, 180, 360, 500)
        both = [
            _det(7, A_SEP, (100.0, 80.0)),
            _det(88, neighbor, (140.0, 80.0)),  # 40cm: old merge swallowed this
        ]
        frame = _frame((A_SEP, BLUE), (neighbor, RED))
        mapper.apply(both, 1, frame)
        out = mapper.apply(both, 2, frame)
        by_raw = {d.get("raw_track_id"): d.get("track_id") for d in out}
        self.assertEqual(by_raw.get(7), 1, f"lost ID1: {out}")
        self.assertEqual(by_raw.get(88), 2, f"neighbor stayed unlabeled: {out}")

    def test_far_duplicate_of_occupied_id_is_hidden(self) -> None:
        """A split box of an ID already on screen is dropped, not shown as person."""
        mapper = StableIdMapper(min_hits=1, encoder=None, gallery_dir=None)
        desk = [_det(7, A_SEP, (100.0, 80.0))]
        frame = _frame((A_SEP, BLUE))
        mapper.apply(desk, 1, frame)

        both = [
            _det(7, A_SEP, (100.0, 80.0)),
            _det(88, B_SEP, (700.0, 500.0)),
        ]
        frame2 = _frame((A_SEP, BLUE), (B_SEP, BLUE))
        out = mapper.apply(both, 2, frame2)
        by_raw = {d.get("raw_track_id"): d.get("track_id") for d in out}
        self.assertEqual(by_raw.get(7), 1)
        self.assertNotIn(88, by_raw, f"duplicate box still drawn: {out}")
        self.assertNotIn(2, _ids(out))

    def test_id_plus_person_nested_split_hides_person(self) -> None:
        """An IDed body plus a nested unlabeled fragment must not show both."""
        mapper = StableIdMapper(min_hits=1, encoder=None, gallery_dir=None)
        big = A_SEP
        frag = (100, 220, 180, 420)  # nested inside A_SEP
        mapper.apply([_det(7, big, (100.0, 80.0))], 1, _frame((big, BLUE)))
        out = mapper.apply(
            [
                _det(7, big, (100.0, 80.0)),
                _det(88, frag, (105.0, 85.0)),
            ],
            2,
            _frame((big, BLUE), (frag, BLUE)),
        )
        by_raw = {d.get("raw_track_id"): d.get("track_id") for d in out}
        self.assertEqual(by_raw.get(7), 1)
        self.assertNotIn(88, by_raw, f"ID and person both drawn: {out}")
        self.assertEqual(len(out), 1)

    def test_far_overlapping_walker_does_not_inherit_id1_via_collapse(
        self,
    ) -> None:
        """test4 ~58s: edge walker overlaps seated ID1 but must not sticky-bind."""
        mapper = StableIdMapper(min_hits=1, encoder=None, gallery_dir=None)
        # Fit inside the 1280x720 synthetic frame used by these tests.
        seat = (900, 180, 1050, 480)
        walker = (920, 100, 1270, 700)  # overlaps seat; far world point
        mapper.apply(
            [_det(7, seat, (490.0, 328.0))], 1, _frame((seat, BLUE))
        )
        out = mapper.apply(
            [
                _det(7, seat, (490.0, 328.0)),
                _det(99, walker, (375.0, 530.0)),
            ],
            2,
            _frame((seat, BLUE), (walker, RED)),
        )
        by_raw = {d.get("raw_track_id"): d.get("track_id") for d in out}
        self.assertEqual(by_raw.get(7), 1)
        # Walker may still be drawn as person, but must not own ID1.
        self.assertNotEqual(by_raw.get(99), 1, f"walker took ID1: {out}")
        self.assertNotEqual(
            mapper._raw_to_stable.get(99), 1, f"walker raw bound to ID1: {out}"
        )
        # Next frame: walker alone must not keep ID1 via sticky raw.
        alone = mapper.apply(
            [_det(99, walker, (370.0, 520.0))],
            3,
            _frame((walker, RED)),
        )
        alone_raw = {d.get("raw_track_id"): d.get("track_id") for d in alone}
        self.assertNotEqual(
            alone_raw.get(99), 1, f"next frame teleported ID1: {alone}"
        )

    def test_nearby_same_clothes_classmate_keeps_a_box(self) -> None:
        """A classmate at the next seat must not be hidden as ID1's ghost."""
        mapper = StableIdMapper(min_hits=1, encoder=None, gallery_dir=None)
        neighbor = (240, 180, 360, 500)
        both = [
            _det(7, A_SEP, (100.0, 80.0)),
            _det(88, neighbor, (160.0, 80.0)),
        ]
        frame = _frame((A_SEP, BLUE), (neighbor, BLUE))
        mapper.apply(both, 1, frame)
        out = mapper.apply(both, 2, frame)
        by_raw = {d.get("raw_track_id"): d.get("track_id") for d in out}
        self.assertEqual(by_raw.get(7), 1)
        self.assertIn(88, by_raw, f"classmate box dropped: {out}")

    def test_crowd_light_overlap_keeps_both_boxes(self) -> None:
        """Seated classmates with IoU ~0.20 must both stay visible."""
        mapper = StableIdMapper(min_hits=1, encoder=None, gallery_dir=None)
        a = A_SEP
        b = (160, 180, 280, 500)
        both = [
            _det(7, a, (100.0, 80.0)),
            _det(88, b, (140.0, 80.0)),
        ]
        frame = _frame((a, BLUE), (b, BLUE))
        mapper.apply(both, 1, frame)
        out = mapper.apply(both, 2, frame)
        by_raw = {d.get("raw_track_id"): d.get("track_id") for d in out}
        self.assertIn(7, by_raw, f"lost left box: {out}")
        self.assertIn(88, by_raw, f"crowd overlap dropped a person: {out}")

    def test_hold_missing_survives_crowd_mega_box(self) -> None:
        """A YOLO blob covering several people must not erase a known ID."""
        mapper = StableIdMapper(min_hits=1, encoder=None, gallery_dir=None)
        small = A_SEP
        mega = (40, 80, 520, 560)
        mapper._last_out = [
            {
                **_det(7, small, (100.0, 80.0)),
                "raw_track_id": 7,
                "track_id": 1,
            }
        ]
        mapper._last_out_frame = 1
        mapper._sid_last_real[1] = 1
        current = [
            {
                **_det(9, mega, (300.0, 90.0)),
                "raw_track_id": 9,
                "track_id": 2,
            }
        ]
        out = mapper._hold_missing_interior(current, 2, _frame())
        sids = _ids(out)
        self.assertIn(1, sids, f"mega-box swallowed ID1: {out}")
        self.assertGreaterEqual(len(out), 2)

    def test_clear_newcomer_still_gets_its_own_id(self) -> None:
        """Hiding split boxes must not starve a real second person of an ID."""
        mapper = StableIdMapper(min_hits=1, encoder=None, gallery_dir=None)
        desk = [_det(7, A_SEP, (100.0, 80.0))]
        mapper.apply(desk, 1, _frame((A_SEP, BLUE)))

        both = [
            _det(7, A_SEP, (100.0, 80.0)),
            _det(88, B_SEP, (700.0, 500.0)),
        ]
        frame2 = _frame((A_SEP, BLUE), (B_SEP, RED))
        out = mapper.apply(both, 2, frame2)
        by_raw = {d.get("raw_track_id"): d.get("track_id") for d in out}
        self.assertEqual(by_raw.get(7), 1)
        self.assertEqual(by_raw.get(88), 2, f"newcomer got no ID: {out}")

    def test_pending_unknown_cannot_bypass_duplicate_guard(self) -> None:
        """Expired quarantine must rejoin normal matching, never mint directly."""
        mapper = StableIdMapper(min_hits=1, encoder=None, gallery_dir=None)
        desk = [_det(7, A_SEP, (100.0, 80.0))]
        frame = _frame((A_SEP, BLUE))
        mapper.apply(desk, 1, frame)

        # One count before the old direct-allocation threshold.
        mapper._unknown_raw_pending[88] = (2, 1)
        both = [
            _det(7, A_SEP, (100.0, 80.0)),
            _det(88, B_SEP, (700.0, 500.0)),
        ]
        frame2 = _frame((A_SEP, BLUE), (B_SEP, BLUE))
        out = mapper.apply(both, 2, frame2)
        by_raw = {d.get("raw_track_id"): d.get("track_id") for d in out}
        self.assertEqual(by_raw.get(7), 1)
        self.assertNotIn(88, by_raw, f"pending path drew duplicate: {out}")
        self.assertNotIn(2, _ids(out))

    def test_latest_color_reclaims_after_pose_change(self) -> None:
        mapper = StableIdMapper(min_hits=1, encoder=None, gallery_dir=None)
        desk = [_det(7, A_SEP, (100.0, 80.0))]
        frame_stand = _frame((A_SEP, BLUE))
        mapper.apply(desk, 1, frame_stand)
        frame_sit = _frame((A_SEP, RED))
        sit_color = mapper._color_feat_from_crop(
            mapper._crop_person(frame_sit, A_SEP)
        )
        mapper._remember_latest_color(1, sit_color)
        self.assertGreater(mapper._best_color_sim(sit_color, 1), 0.90)
        self.assertEqual(mapper._color_reclaim(sit_color, set()), 1)
        self.assertLess(
            mapper._appear_sim(
                mapper._gallery_first_color[1], sit_color
            ),
            0.55,
        )

    def test_far_same_clothes_cannot_steal_vacant_id(self) -> None:
        """RTSP ID3: two dark shirts at different desks must not ping-pong."""
        mapper = StableIdMapper(
            min_hits=1, encoder=None, gallery_dir=None, fps=20.0
        )
        mapper.apply(
            [_det(7, A_SEP, (100.0, 80.0))], 1, _frame((A_SEP, BLUE))
        )
        out = mapper.apply(
            [_det(88, B_SEP, (700.0, 500.0))],
            40,
            _frame((B_SEP, BLUE)),
        )
        by_raw = {d.get("raw_track_id"): d.get("track_id") for d in out}
        self.assertNotEqual(by_raw.get(88), 1, f"far classmate stole ID1: {out}")

    def test_color_reclaim_ignores_far_same_clothes(self) -> None:
        mapper = StableIdMapper(min_hits=1, encoder=None, gallery_dir=None)
        mapper.apply(
            [_det(7, A_SEP, (100.0, 80.0))], 1, _frame((A_SEP, BLUE))
        )
        frame = _frame((B_SEP, BLUE))
        color = mapper._color_feat_from_crop(mapper._crop_person(frame, B_SEP))
        self.assertIsNone(
            mapper._color_reclaim(color, set(), world=(700.0, 500.0))
        )
        self.assertEqual(
            mapper._color_reclaim(color, set(), world=(120.0, 90.0)),
            1,
        )

    def test_enrollment_anchor_is_not_ema_washed(self) -> None:
        mapper = StableIdMapper(min_hits=1, encoder=None, gallery_dir=None)
        feat = mapper.appearance_feat(_frame((A_SEP, BLUE)), A_SEP)
        self.assertIsNotNone(feat)
        meta = {"feats": [feat.copy()], "feat": feat.copy()}
        mapper._update_prototypes(meta, feat, allow_new=True)
        self.assertTrue(np.allclose(meta["feats"][0], feat))
        self.assertGreaterEqual(len(meta["feats"]), 1)

    def test_far_unlike_look_restores_anchor_and_drops_id(self) -> None:
        """Polluted extra prototype must not keep ID1 on a far, different person."""
        mapper = StableIdMapper(
            min_hits=1, encoder=None, gallery_dir=None, fps=20.0
        )
        mapper.apply(
            [_det(7, A_SEP, (100.0, 80.0))], 1, _frame((A_SEP, BLUE))
        )
        first = mapper._gallery_first_feat[1]
        thief = mapper.appearance_feat(_frame((B_SEP, RED)), B_SEP)
        self.assertIsNotNone(first)
        self.assertIsNotNone(thief)
        self.assertLess(mapper._appear_sim(first, thief), mapper.appear_thresh)
        mapper._stable[1]["feats"] = [first.copy(), thief.copy()]
        out = mapper.apply(
            [_det(7, B_SEP, (700.0, 500.0))],
            20,
            _frame((B_SEP, RED)),
        )
        protos = mapper._proto_list(mapper._stable[1])
        self.assertEqual(len(protos), 1)
        self.assertGreater(mapper._appear_sim(protos[0], first), 0.95)
        by_raw = {d.get("raw_track_id"): d.get("track_id") for d in out}
        self.assertNotEqual(by_raw.get(7), 1, f"polluted ID1 followed the thief: {out}")

    def test_crossing_crowd_does_not_restore(self) -> None:
        """Overlapping people keep sticky IDs; do not restore-loop."""
        mapper = StableIdMapper(
            min_hits=1, encoder=None, gallery_dir=None, fps=20.0
        )
        mapper.apply(
            [_det(7, A_SEP, (100.0, 80.0))], 1, _frame((A_SEP, BLUE))
        )
        first = mapper._gallery_first_feat[1]
        thief = mapper.appearance_feat(_frame((B_SEP, RED)), B_SEP)
        self.assertIsNotNone(first)
        self.assertIsNotNone(thief)
        mapper._stable[1]["feats"] = [first.copy(), thief.copy()]
        # Overlap at the *same* desk — sticky raw must keep ID1; restore must
        # not fire just because the crop briefly looks unlike enrollment.
        crowd = [
            _det(7, A_OVER, (120.0, 80.0)),
            _det(22, B_OVER, (140.0, 80.0)),
        ]
        out = mapper.apply(
            crowd, 20, _frame((A_OVER, RED), (B_OVER, BLUE))
        )
        by_raw = {d.get("raw_track_id"): d.get("track_id") for d in out}
        self.assertEqual(
            by_raw.get(7), 1, f"crossing restored ID1 off raw7: {out}"
        )

    def test_two_visible_people_do_not_restore(self) -> None:
        """Restore is only for a lone far unlike crop, not a crowded frame."""
        mapper = StableIdMapper(
            min_hits=1, encoder=None, gallery_dir=None, fps=20.0
        )
        mapper.apply(
            [_det(7, A_SEP, (100.0, 80.0))], 1, _frame((A_SEP, BLUE))
        )
        thief = mapper.appearance_feat(_frame((B_SEP, RED)), B_SEP)
        self.assertIsNotNone(thief)
        crowd = [
            _det(7, B_SEP, (700.0, 500.0)),
            _det(88, A_SEP, (100.0, 80.0)),
        ]
        self.assertFalse(
            mapper._should_restore_enrollment(
                1, thief, (700.0, 500.0), frame_idx=20, work=crowd
            )
        )
        self.assertTrue(
            mapper._should_restore_enrollment(
                1,
                thief,
                (700.0, 500.0),
                frame_idx=20,
                work=[_det(7, B_SEP, (700.0, 500.0))],
            )
        )

    def test_separated_desks_are_not_a_crossing_scene(self) -> None:
        mapper = StableIdMapper(min_hits=1, encoder=None, gallery_dir=None)
        far = [
            _det(7, A_SEP, (100.0, 80.0)),
            _det(88, B_SEP, (700.0, 500.0)),
        ]
        self.assertFalse(mapper._is_crossing_scene(far))
        over = [
            _det(7, A_OVER, (120.0, 80.0)),
            _det(22, B_OVER, (140.0, 80.0)),
        ]
        self.assertTrue(mapper._is_crossing_scene(over))


class ReviewDumpGridTests(unittest.TestCase):
    def test_saves_on_fixed_tens_not_shifted_by_a_miss(self) -> None:
        self.assertFalse(review_frame_due(196, 10, -10**9))
        self.assertTrue(review_frame_due(200, 10, -10**9))
        self.assertFalse(review_frame_due(201, 10, 200))
        self.assertFalse(review_frame_due(211, 10, 200))
        self.assertTrue(review_frame_due(210, 10, 200))
        self.assertTrue(review_frame_due(220, 10, 200))


if __name__ == "__main__":
    unittest.main()

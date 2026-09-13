"""Enrollment gate: overlapping boxes must not mint Stable-IDs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from stable_id import StableIdMapper

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

    def test_split_outfit_crop_does_not_enroll(self) -> None:
        mapper = StableIdMapper(min_hits=1, encoder=None, gallery_dir=None)
        box = A_SEP
        frame = np.full((H, W, 3), 28, dtype=np.uint8)
        x1, y1, x2, y2 = box
        mid = (x1 + x2) // 2
        frame[y1:y2, x1:mid] = (20, 40, 210)
        frame[y1:y2, mid:x2] = (210, 40, 20)
        self.assertFalse(mapper._can_enroll_new_id(frame, box, 0.85, []))


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


if __name__ == "__main__":
    unittest.main()

"""PoseTrackHistory: mode stickiness + foot EMA reduce left-desk drift."""

from __future__ import annotations

import unittest

import numpy as np

from detect_grid import PoseTrackHistory


def _kpts(
    hips=(100.0, 200.0),
    shoulders=(100.0, 120.0),
    ankles=(100.0, 360.0),
    knees=(100.0, 280.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Minimal COCO-17 keypoints (zeros elsewhere)."""
    xy = np.zeros((17, 2), dtype=np.float32)
    conf = np.zeros(17, dtype=np.float32)
    # shoulders 5/6, hips 11/12, knees 13/14, ankles 15/16
    for i, pt in ((5, shoulders), (6, shoulders), (11, hips), (12, hips)):
        xy[i] = pt
        conf[i] = 0.9
    for i, pt in ((13, knees), (14, knees), (15, ankles), (16, ankles)):
        xy[i] = pt
        conf[i] = 0.9
    return xy, conf


class PoseFootSmoothTests(unittest.TestCase):
    def test_mode_needs_confirm_before_switch(self) -> None:
        hist = PoseTrackHistory(sit_confirm=1, mode_confirm=2, foot_alpha=1.0)
        # Standing near frame bottom with full legs → pose
        xy, conf = _kpts(
            hips=(100.0, 480.0),
            shoulders=(100.0, 400.0),
            knees=(100.0, 560.0),
            ankles=(100.0, 640.0),
        )
        f1, m1, _ = hist.resolve(1, xy, conf, 50, 360, 150, 650, 800, 0.35)
        self.assertEqual(m1, "pose")
        self.assertIsNotNone(f1)

        # Table-cut torso: hips near box bottom, no lower joints → stand_drop.
        xy2, conf2 = _kpts(
            hips=(100.0, 560.0),
            shoulders=(100.0, 480.0),
            knees=(100.0, 600.0),
            ankles=(100.0, 640.0),
        )
        for i in (13, 14, 15, 16):
            conf2[i] = 0.0
        hist._tracks[1]["full_hits"] = 3
        hist._tracks[1]["history_age"] = 0
        hist._tracks[1]["torso"] = 80.0
        hist._tracks[1]["hip_to_foot"] = (0.0, 160.0)
        f2, m2, _ = hist.resolve(1, xy2, conf2, 50, 400, 150, 620, 800, 0.35)
        self.assertEqual(m2, "pose")  # still sticky
        self.assertIsNotNone(f2)

        f3, m3, _ = hist.resolve(1, xy2, conf2, 50, 400, 150, 620, 800, 0.35)
        self.assertEqual(m3, "stand_drop")

    def test_keeps_last_foot_when_pose_unsure(self) -> None:
        hist = PoseTrackHistory(sit_confirm=3, mode_confirm=1, foot_alpha=1.0)
        xy, conf = _kpts()
        f1, m1, _ = hist.resolve(7, xy, conf, 50, 80, 150, 370, 800, 0.35)
        self.assertIsNotNone(f1)
        # No joints → finalize should reuse last foot instead of None.
        blank = np.zeros((17, 2), dtype=np.float32)
        blank_c = np.zeros(17, dtype=np.float32)
        f2, m2, _ = hist.resolve(7, blank, blank_c, 50, 80, 150, 370, 800, 0.35)
        self.assertEqual(f2, f1)
        self.assertEqual(m2, m1)


if __name__ == "__main__":
    unittest.main()

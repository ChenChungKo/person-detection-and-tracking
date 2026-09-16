"""Per-person sticky floor cells: light immediately, no adjacent flicker."""

from __future__ import annotations

import unittest

from grid_occupancy import CellStabilizer


def _dets(*pairs: tuple[int, tuple[int, int] | None]) -> list[dict]:
    return [{"track_id": tid, "cell": cell} for tid, cell in pairs]


class CellStabilizerTests(unittest.TestCase):
    def test_lights_on_first_valid_cell(self) -> None:
        stab = CellStabilizer(hold=2)
        occ = stab.update(_dets((1, (3, 4))))
        self.assertEqual(occ, {(3, 4): [1]})

    def test_adjacent_hops_keep_first_cell(self) -> None:
        stab = CellStabilizer(hold=2)
        stab.update(_dets((1, (3, 4))))
        self.assertEqual(stab.update(_dets((1, (4, 4)))), {(3, 4): [1]})
        self.assertEqual(stab.update(_dets((1, (3, 4)))), {(3, 4): [1]})
        self.assertEqual(stab.update(_dets((1, (4, 4)))), {(3, 4): [1]})

    def test_two_consecutive_neighbor_hits_switch_cell(self) -> None:
        stab = CellStabilizer(hold=2)
        stab.update(_dets((1, (3, 4))))
        stab.update(_dets((1, (4, 4))))
        self.assertEqual(stab.update(_dets((1, (4, 4)))), {(4, 4): [1]})

    def test_non_adjacent_step_switches_immediately(self) -> None:
        stab = CellStabilizer(hold=2)
        stab.update(_dets((1, (3, 4))))
        self.assertEqual(stab.update(_dets((1, (5, 6)))), {(5, 6): [1]})

    def test_one_miss_keeps_cell_two_misses_clear(self) -> None:
        stab = CellStabilizer(hold=2)
        stab.update(_dets((1, (3, 4))))
        self.assertEqual(stab.update(_dets((1, None))), {(3, 4): [1]})
        self.assertEqual(stab.update(_dets((1, None))), {})

    def test_hold_one_follows_every_adjacent_hop(self) -> None:
        stab = CellStabilizer(hold=1)
        stab.update(_dets((1, (3, 4))))
        self.assertEqual(stab.update(_dets((1, (4, 4)))), {(4, 4): [1]})


if __name__ == "__main__":
    unittest.main()

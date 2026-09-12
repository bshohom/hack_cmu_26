"""Generic mesh-aware BC node selection. No task-specific fixtures."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class MeshAwareBcTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            from to_agent.contracts import BoxRegion
            from to_agent.meshing.masks import select_nodes_for_region
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"to_agent unavailable: {exc}") from exc
        cls.BoxRegion = BoxRegion
        cls.select = staticmethod(select_nodes_for_region)

    def _grid(self, h: float = 6.0) -> np.ndarray:
        xs = np.arange(0.0, 30.0 + 0.5 * h, h)
        ys = np.arange(-12.0, 12.0 + 0.5 * h, h)
        zs = np.arange(0.0, 24.0 + 0.5 * h, h)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    def test_zero_thickness_face_between_planes_selects_nearest_nodes(self) -> None:
        nodes = self._grid(h=6.0)
        # Face at x=0.1 sits between x=0 and x=6 grid planes.
        region = self.BoxRegion(min=(0.1, -8.0, 2.0), max=(0.1, 8.0, 16.0))
        exact = np.all((nodes >= (0.1, -8.0, 2.0)) & (nodes <= (0.1, 8.0, 16.0)), axis=1)
        self.assertFalse(exact.any())
        sel = self.select(region, nodes, h=6.0)
        self.assertTrue(sel.any())
        chosen = nodes[sel]
        self.assertTrue(np.all(np.abs(chosen[:, 0] - 0.1) <= 3.0))
        self.assertTrue(np.all((chosen[:, 1] >= -8.0) & (chosen[:, 1] <= 8.0)))
        self.assertTrue(np.all((chosen[:, 2] >= 2.0) & (chosen[:, 2] <= 16.0)))

    def test_exact_occupancy_is_unchanged(self) -> None:
        nodes = self._grid(h=6.0)
        region = self.BoxRegion(min=(-1.0, -12.0, -1.0), max=(1.0, 12.0, 24.0))
        sel = self.select(region, nodes, h=6.0)
        self.assertTrue(sel.any())
        self.assertTrue(np.all(np.abs(nodes[sel][:, 0]) <= 1.0))

    def test_unresolvable_region_fails_closed(self) -> None:
        nodes = self._grid(h=6.0)
        region = self.BoxRegion(min=(100.0, 100.0, 100.0), max=(100.2, 100.2, 100.2))
        sel = self.select(region, nodes, h=6.0)
        self.assertFalse(sel.any())


if __name__ == "__main__":
    unittest.main()

import numpy as np

from to_agent.contracts import (
    BoxRegion,
    CylinderRegion,
    DifferenceRegion,
    HalfSpaceRegion,
    SphereRegion,
    UnionRegion,
)
from to_agent.regions import bounds, contains

PTS = np.array([[0, 0, 0], [1, 1, 1], [5, 0, 0], [0, 0, 10]], dtype=float)


def test_box():
    r = BoxRegion(min=(-1, -1, -1), max=(2, 2, 2))
    assert contains(r, PTS).tolist() == [True, True, False, False]


def test_cylinder_disk_and_annulus():
    disk = CylinderRegion(center=(0, 0, 0), axis="z", r_max=2.0, along=(-1, 5))
    assert contains(disk, PTS).tolist() == [True, True, False, False]
    ring = CylinderRegion(center=(0, 0, 0), axis="z", r_min=1.0, r_max=2.0, along=(-1, 5))
    assert contains(ring, PTS).tolist() == [False, True, False, False]
    along_x = CylinderRegion(center=(0, 0, 0), axis="x", r_max=0.5, along=(4, 6))
    assert contains(along_x, PTS).tolist() == [False, False, True, False]


def test_sphere_halfspace():
    assert contains(SphereRegion(center=(0, 0, 0), radius=2.0), PTS).tolist() == [True, True, False, False]
    hs = HalfSpaceRegion(point=(0, 0, 5), normal=(0, 0, 1))
    assert contains(hs, PTS).tolist() == [False, False, False, True]


def test_union_difference_bounds():
    a = BoxRegion(min=(-1, -1, -1), max=(2, 2, 2))
    b = SphereRegion(center=(0, 0, 10), radius=1.0)
    u = UnionRegion(regions=[a, b])
    assert contains(u, PTS).tolist() == [True, True, False, True]
    d = DifferenceRegion(a=a, b=SphereRegion(center=(0, 0, 0), radius=0.5))
    assert contains(d, PTS).tolist() == [False, True, False, False]
    lo, hi = bounds(u)
    assert lo.tolist() == [-1, -1, -1] and hi.tolist() == [2, 2, 11]

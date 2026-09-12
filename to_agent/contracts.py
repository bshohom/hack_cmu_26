"""Agent-facing problem definition.

A TOProblem is plain data (YAML/JSON) built from generic region primitives; nothing here
knows about any particular part. The meshing/solver layers turn it into torch-fem tensors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Iterator, Literal, Union

import yaml
from pydantic import AliasChoices, BaseModel, Field, model_validator

Vec3 = tuple[float, float, float]
Axis = Literal["x", "y", "z"]

# Where a value came from. One vocabulary for the whole pipeline: registration
# measurements, agent-supplied regions, and anything a builder invented.
#   observed  - measured off a reconstruction of the real object
#   user      - stated by the user
#   derived   - computed from observed/user values
#   assumed   - invented by a default because nothing better was available
Provenance = Literal["observed", "user", "derived", "assumed"]

# `confidence` was the original spelling; problem YAML in data/ still uses it.
_PROVENANCE_FIELD = Field(
    default=None, validation_alias=AliasChoices("provenance", "confidence")
)


class Assumption(BaseModel):
    """One invented value that materially affects the result.

    Anything the builder made up lands here so the UI can state it, instead of it
    disappearing into a prose notes string.
    """

    field: str  # what was assumed, e.g. "supports"
    value: str  # what it was set to
    basis: str  # why, e.g. "no fixed_regions supplied; fell back to clamp layout"


# ----------------------------------------------------------------------------- regions
class BoxRegion(BaseModel):
    type: Literal["box"] = "box"
    min: Vec3
    max: Vec3


class CylinderRegion(BaseModel):
    """Annular cylinder around `axis` through `center`.

    Radial band r_min <= r <= r_max (r_min = 0 gives a solid disk); the coordinate along the
    axis must lie in `along` (absolute coordinates, e.g. z0..z1 for axis "z").
    """

    type: Literal["cylinder"] = "cylinder"
    center: Vec3
    axis: Axis = "z"
    r_min: float = 0.0
    r_max: float
    along: tuple[float, float]


class SphereRegion(BaseModel):
    type: Literal["sphere"] = "sphere"
    center: Vec3
    radius: float


class CapsuleRegion(BaseModel):
    """Points within `radius` of the segment a-b: one structural member / rod."""

    type: Literal["capsule"] = "capsule"
    a: Vec3
    b: Vec3
    radius: float


class HalfSpaceRegion(BaseModel):
    """Points p with (p - point) . normal >= 0."""

    type: Literal["halfspace"] = "halfspace"
    point: Vec3
    normal: Vec3


class NearPointsRegion(BaseModel):
    """Points within `tol` of any point of a point cloud file (OBJ/PLY/XYZ)."""

    type: Literal["near_points"] = "near_points"
    path: str
    tol: float


class InsideMeshRegion(BaseModel):
    """Points inside a watertight triangle mesh (STL/OBJ/PLY)."""

    type: Literal["inside_mesh"] = "inside_mesh"
    path: str


class NearMeshRegion(BaseModel):
    """Points within `tol` of a triangle-mesh surface (mesh need not be watertight)."""

    type: Literal["near_mesh"] = "near_mesh"
    path: str
    tol: float


class UnionRegion(BaseModel):
    type: Literal["union"] = "union"
    regions: list[Region]


class IntersectionRegion(BaseModel):
    type: Literal["intersection"] = "intersection"
    regions: list[Region]


class DifferenceRegion(BaseModel):
    """Points in `a` but not in `b`."""

    type: Literal["difference"] = "difference"
    a: Region
    b: Region


Region = Annotated[
    Union[
        BoxRegion,
        CylinderRegion,
        SphereRegion,
        CapsuleRegion,
        HalfSpaceRegion,
        NearPointsRegion,
        InsideMeshRegion,
        NearMeshRegion,
        UnionRegion,
        IntersectionRegion,
        DifferenceRegion,
    ],
    Field(discriminator="type"),
]

UnionRegion.model_rebuild()
IntersectionRegion.model_rebuild()
DifferenceRegion.model_rebuild()


# ----------------------------------------------------------------------------- problem
class Material(BaseModel):
    name: str = "PLA"
    E_MPa: float = 2300.0
    nu: float = 0.35
    density_kg_m3: float | None = None
    yield_MPa: float | None = None
    # Free text describing where these numbers come from (not the Provenance vocabulary:
    # material values are always looked up, never observed or user-measured).
    source_note: str | None = Field(
        default=None, validation_alias=AliasChoices("source_note", "confidence")
    )


class Support(BaseModel):
    id: str
    region: Region
    fixed_dofs: list[Axis] = ["x", "y", "z"]
    provenance: Provenance | None = _PROVENANCE_FIELD


class LoadCase(BaseModel):
    id: str
    region: Region
    force_N: Vec3  # total force; split equally over the nodes selected by `region`
    weight: float = 1.0
    provenance: Provenance | None = _PROVENANCE_FIELD
    # primary: user/agent payload. retention: template stabilization that must
    # stay when agent loads replace the primary set (e.g. tip_retention).
    role: Literal["primary", "retention"] | None = None


class TOProblem(BaseModel):
    units: dict[str, str] = {"length": "mm", "force": "N"}
    material: Material = Field(default_factory=Material)
    # None -> bounding box of the warm-start regions padded by `domain_padding`.
    design_domain: BoxRegion | None = None
    domain_padding: float = 5.0
    warm_start: list[Region] = []
    preserve: list[Region] = []
    void: list[Region] = []
    supports: list[Support]
    load_cases: list[LoadCase]
    safety_factor: float = 2.5
    volume_fraction: float = 0.35
    target_element_size: float = 4.0
    filter_radius: float | None = None  # default: 1.5 * element size
    penal: float = 3.0
    rho_min: float = 0.05
    move: float = 0.2
    max_iters: int = 40
    change_tol: float = 0.01
    notes: str | None = None
    # Values the builder invented rather than took from the agent. Surfaced, not buried.
    assumptions: list[Assumption] = []

    @model_validator(mode="after")
    def _check(self) -> TOProblem:
        if not 0.0 < self.volume_fraction < 1.0:
            raise ValueError("volume_fraction must be in (0, 1)")
        if self.target_element_size <= 0:
            raise ValueError("target_element_size must be positive")
        if not self.supports:
            raise ValueError("at least one support is required")
        if not self.load_cases:
            raise ValueError("at least one load case is required")
        ids = [s.id for s in self.supports] + [lc.id for lc in self.load_cases]
        if len(ids) != len(set(ids)):
            raise ValueError(f"support/load ids must be unique, got {ids}")
        if self.design_domain is None and not self.warm_start:
            raise ValueError("design_domain is required when there is no warm_start")
        return self


# ----------------------------------------------------------------------------- helpers
def iter_regions(problem: TOProblem) -> Iterator[BaseModel]:
    """Yield every region in the problem, recursing into combinators."""

    def walk(r: BaseModel) -> Iterator[BaseModel]:
        yield r
        if isinstance(r, (UnionRegion, IntersectionRegion)):
            for c in r.regions:
                yield from walk(c)
        elif isinstance(r, DifferenceRegion):
            yield from walk(r.a)
            yield from walk(r.b)

    if problem.design_domain is not None:
        yield from walk(problem.design_domain)
    for r in [*problem.warm_start, *problem.preserve, *problem.void]:
        yield from walk(r)
    for s in problem.supports:
        yield from walk(s.region)
    for lc in problem.load_cases:
        yield from walk(lc.region)


def resolve_paths(problem: TOProblem, base_dir: Path) -> None:
    """Make file paths inside regions absolute relative to `base_dir` (in place)."""
    for r in iter_regions(problem):
        path = getattr(r, "path", None)
        if path is not None and not Path(path).is_absolute():
            r.path = str((base_dir / path).resolve())


def load_problem(path: str | Path) -> TOProblem:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    problem = TOProblem.model_validate(data)
    resolve_paths(problem, path.parent)
    return problem


def save_problem(problem: TOProblem, path: str | Path, header: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(problem.model_dump(mode="json"), sort_keys=False)
    if header:
        text = "".join(f"# {line}\n" for line in header.splitlines()) + text
    path.write_text(text)

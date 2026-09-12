"""Part-specific problem builders, keyed by the candidate's `task`."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..contracts import TOProblem
from .cupholder import build_cupholder_problem
from .hook import build_hook_problem
from .shelf import build_shelf_problem

# builder(dims_path, points_path, element_size, volume_fraction, safety_factor, ...) -> (TOProblem, report)
Builder = Callable[..., tuple[TOProblem, dict]]

BUILDERS: dict[str, Builder] = {
    "cupholder": build_cupholder_problem,
    "desk_bag_hook": build_hook_problem,
    "stapler_shelf": build_shelf_problem,
}

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Default input triples for the demo tasks (repo root).
DEMO_FILES: dict[str, tuple[Path, Path]] = {
    "cupholder": (REPO_ROOT / "cupholder_dimensions.txt", REPO_ROOT / "cupholder_surface_particles.obj"),
    "desk_bag_hook": (
        REPO_ROOT / "desk_bag_hook_5kg_100mm_final_dimensions.txt",
        REPO_ROOT / "desk_bag_hook_5kg_100mm_final_particles.obj",
    ),
    "stapler_shelf": (
        REPO_ROOT / "stapler_shelf_100mm_guaranteed_flat_top_dimensions.txt",
        REPO_ROOT / "stapler_shelf_100mm_guaranteed_flat_top_particles.obj",
    ),
}


def get_builder(task: str) -> Builder:
    try:
        return BUILDERS[task]
    except KeyError:
        raise KeyError(f"no problem builder for task {task!r}; known: {sorted(BUILDERS)}") from None


def candidate_registration(task: str, dims_path: str | Path, points_path: str | Path) -> dict | None:
    """Imported-demo measurement: plane transform + named features in desk_edge_frame.

    Coordinates live in the part-specific demo module, not in the adapter or orchestrator.
    """
    if task == "desk_bag_hook":
        from .hook import hook_registration

        return hook_registration(dims_path, points_path)
    return None

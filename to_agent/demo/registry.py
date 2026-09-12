"""Part-specific problem builders, keyed by the candidate's `task`."""

from __future__ import annotations

from typing import Callable

from ..contracts import TOProblem
from .cupholder import build_cupholder_problem

# builder(dims_path, points_path, element_size, volume_fraction, safety_factor) -> (TOProblem, report)
Builder = Callable[..., tuple[TOProblem, dict]]

BUILDERS: dict[str, Builder] = {
    "cupholder": build_cupholder_problem,
}


def get_builder(task: str) -> Builder:
    try:
        return BUILDERS[task]
    except KeyError:
        raise KeyError(f"no problem builder for task {task!r}; known: {sorted(BUILDERS)}") from None

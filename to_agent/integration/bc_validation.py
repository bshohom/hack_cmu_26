"""Required support/load usability against the final non-void domain.

Nodal BCs are applied to nodes, so usability is "this node is incident to at least
one non-void element", not element-centroid overlap with the BC box.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..regions import contains


def _usable_nodes(mesh, void: np.ndarray) -> np.ndarray:
    elements = mesh.elements.numpy() if hasattr(mesh.elements, "numpy") else np.asarray(mesh.elements)
    usable = np.zeros(mesh.n_nodes, dtype=bool)
    usable[elements[~void].reshape(-1)] = True
    return usable


def validate_required_bcs(problem, mesh, void: np.ndarray) -> dict[str, Any]:
    """Per-region usable-node counts on the final void mask (envelope included)."""
    nodes = mesh.nodes.numpy()
    centroids = mesh.centroids
    usable = _usable_nodes(mesh, void)
    regions: list[dict[str, Any]] = []
    hard = False
    for kind, item in (
        *[("support", s) for s in problem.supports],
        *[("load", c) for c in problem.load_cases],
    ):
        node_sel = contains(item.region, nodes)
        elem_sel = contains(item.region, centroids)
        n_nodes = int(node_sel.sum())
        n_usable = int((node_sel & usable).sum())
        n_unusable = n_nodes - n_usable
        frac = (n_usable / n_nodes) if n_nodes else 0.0
        if n_nodes == 0 or n_usable == 0:
            status = "hard_infeasible"
            hard = True
        elif n_unusable:
            status = "partial"
        else:
            status = "ok"
        regions.append({
            "kind": kind,
            "id": item.id,
            "requested_nodes": n_nodes,
            "requested_elements": int(elem_sel.sum()),
            "usable_nodes": n_usable,
            "unusable_nodes": n_unusable,
            "usable_fraction": round(float(frac), 4),
            "status": status,
        })
    return {"regions": regions, "hard_infeasible": hard}

"""Structured hexahedral grid over a box design domain, via torchfem.mesh.cube_hexa."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import device as _device  # noqa: F401  (sets default dtype, guards CUDA env)
import torch
from torchfem.mesh import cube_hexa

from ..contracts import BoxRegion


@dataclass
class HexMesh:
    nodes: torch.Tensor  # [n_nodes, 3] (cpu)
    elements: torch.Tensor  # [n_elem, 8] (cpu, long)
    centroids: np.ndarray  # [n_elem, 3]
    origin: np.ndarray  # (3,) min corner
    spacing: np.ndarray  # (3,) element size per axis (<= target)
    shape: tuple[int, int, int]  # elements per axis
    elem_ijk: np.ndarray  # [n_elem, 3] grid index of each element
    node_ijk: np.ndarray  # [n_nodes, 3] grid index of each node

    @property
    def n_elem(self) -> int:
        return int(self.elements.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_dof(self) -> int:
        return 3 * self.n_nodes

    @property
    def h(self) -> float:
        return float(self.spacing.max())

    @property
    def elem_volume(self) -> float:
        return float(np.prod(self.spacing))


def grid_shape(domain: BoxRegion, target_h: float) -> tuple[np.ndarray, np.ndarray]:
    """Elements per axis and the resulting spacing for a target element size."""
    lo, hi = np.asarray(domain.min, float), np.asarray(domain.max, float)
    L = hi - lo
    if np.any(L <= 0):
        raise ValueError(f"design_domain has non-positive extent: min={lo}, max={hi}")
    n = np.maximum(1, np.ceil(L / target_h - 1e-9)).astype(int)
    return n, L / n


def build_hex_grid(domain: BoxRegion, target_h: float) -> HexMesh:
    lo = np.asarray(domain.min, float)
    n, spacing = grid_shape(domain, target_h)
    L = spacing * n
    nodes, elements = cube_hexa(int(n[0]) + 1, int(n[1]) + 1, int(n[2]) + 1, float(L[0]), float(L[1]), float(L[2]))
    nodes = nodes.to(torch.get_default_dtype()) + torch.as_tensor(lo, dtype=torch.get_default_dtype())
    elements = elements.long()
    centroids = nodes[elements].mean(dim=1).numpy()
    elem_ijk = np.clip(np.floor((centroids - lo) / spacing).astype(int), 0, n - 1)
    node_ijk = np.clip(np.rint((nodes.numpy() - lo) / spacing).astype(int), 0, n)
    return HexMesh(
        nodes=nodes,
        elements=elements,
        centroids=centroids,
        origin=lo,
        spacing=spacing,
        shape=(int(n[0]), int(n[1]), int(n[2])),
        elem_ijk=elem_ijk,
        node_ijk=node_ijk,
    )

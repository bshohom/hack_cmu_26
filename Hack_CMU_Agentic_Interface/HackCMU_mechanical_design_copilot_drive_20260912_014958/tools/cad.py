"""CAD export tool interface.

Consumes structured geometry / topology. Does not invent mesh vertices in an LLM.
When the topology stage produced a real STL (to_agent isosurface export), that file is
the printable output; otherwise the placeholder below is returned.
"""

from __future__ import annotations

from pathlib import Path

from schemas import CadInput, CadOutput


def generate_cad(inp: CadInput) -> CadOutput:
    """Printable export: pass through the optimized STL when it exists.

    TODO(Aman): optional later hook if CAD export shares the registration frame.
    """
    ref = Path(inp.topology.optimized_geometry_ref)
    if not inp.topology.is_mock and ref.suffix.lower() in {".stl", ".obj"} and ref.exists():
        return CadOutput(
            is_mock=False,
            format=ref.suffix.lower().lstrip("."),
            filename=str(ref),
            notes=(
                "Density isosurface (rho >= 0.5) exported by to_agent from the topology "
                "optimization. Mesh vertices are not LLM-generated. "
                f"{inp.topology.notes}"
            ),
        )
    return CadOutput(
        is_mock=True,
        format="stl",
        filename="cup_holder.stl",
        notes=(
            "Placeholder CAD export from structured topology/geometry. "
            "Mesh vertices are not LLM-generated."
        ),
    )

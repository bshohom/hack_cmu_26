"""CAD export tool interface.

Consumes structured geometry / topology. Does not invent mesh vertices in an LLM.
"""

from __future__ import annotations

from schemas import CadInput, CadOutput


def generate_cad(inp: CadInput) -> CadOutput:
    """Placeholder printable export.

    TODO(Aman): optional later hook if CAD export shares the registration frame.
    """
    return CadOutput(
        is_mock=True,
        format="stl",
        filename="cup_holder.stl",
        notes=(
            "Placeholder CAD export from structured topology/geometry. "
            "Mesh vertices are not LLM-generated."
        ),
    )

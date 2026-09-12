"""Material presets by name — as-printed FDM values, not bulk datasheet numbers.

Numbers are typical manufacturer TDS / literature values for printed test bars at 100 %
infill: modulus in the XY plane, and the design yield taken in the weak Z (layer-adhesion)
direction so an orientation-agnostic check stays conservative. Density is the solid value;
the optimizer removes material explicitly instead of relying on infill.

  PLA  : E ≈ 2300 MPa, strength XY ≈ 45–50 MPa, Z ≈ 30 MPa  -> yield 30
  PETG : E ≈ 1900 MPa, strength XY ≈ 30–35 MPa, Z ≈ 22 MPa  -> yield 22
  ABS  : E ≈ 1900 MPa, strength XY ≈ 30 MPa,    Z ≈ 18 MPa  -> yield 18
  ASA  : E ≈ 1900 MPa, strength XY ≈ 32 MPa,    Z ≈ 20 MPa  -> yield 20
  Nylon: E ≈ 1400 MPa, strength XY ≈ 35 MPa,    Z ≈ 25 MPa  -> yield 25
"""

from __future__ import annotations

from ..contracts import Material

PRINTED = "as-printed FDM values (XY modulus, Z-direction yield)"

PRESETS: dict[str, Material] = {
    "pla": Material(name="PLA (FDM printed)", E_MPa=2300.0, nu=0.35, density_kg_m3=1240.0, yield_MPa=30.0, confidence=PRINTED),
    "petg": Material(name="PETG (FDM printed)", E_MPa=1900.0, nu=0.37, density_kg_m3=1270.0, yield_MPa=22.0, confidence=PRINTED),
    "abs": Material(name="ABS (FDM printed)", E_MPa=1900.0, nu=0.35, density_kg_m3=1040.0, yield_MPa=18.0, confidence=PRINTED),
    "asa": Material(name="ASA (FDM printed)", E_MPa=1900.0, nu=0.35, density_kg_m3=1070.0, yield_MPa=20.0, confidence=PRINTED),
    "nylon": Material(name="Nylon (FDM printed)", E_MPa=1400.0, nu=0.40, density_kg_m3=1130.0, yield_MPa=25.0, confidence=PRINTED),
}


def printed_material(name: str | None = "PLA") -> Material:
    """Preset by name; unknown names fall back to PLA (see material_from_name for the note)."""
    return material_from_name(name)[0]


def material_from_name(name: str | None) -> tuple[Material, str]:
    """Return (material, note). Unknown names fall back to PLA with an explicit note."""
    key = (name or "PLA").strip().lower().replace("-", "").replace(" ", "")
    for alias, target in (("polylactic", "pla"), ("nylon", "nylon"), ("pa12", "nylon"), ("pa6", "nylon")):
        if alias in key:
            key = target
    if key in PRESETS:
        return PRESETS[key].model_copy(), ""
    m = PRESETS["pla"].model_copy()
    m.confidence = f"assumed: '{name}' unknown, PLA printed values used"
    return m, f"material '{name}' unknown; using PLA printed properties"

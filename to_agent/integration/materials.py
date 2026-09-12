"""Material presets by name (FDM-printed thermoplastics, MPa / kg m^-3)."""

from __future__ import annotations

from ..contracts import Material

PRESETS: dict[str, Material] = {
    "pla": Material(name="PLA", E_MPa=2300.0, nu=0.35, density_kg_m3=1240.0, yield_MPa=50.0),
    "petg": Material(name="PETG", E_MPa=2000.0, nu=0.37, density_kg_m3=1270.0, yield_MPa=45.0),
    "abs": Material(name="ABS", E_MPa=2000.0, nu=0.35, density_kg_m3=1040.0, yield_MPa=40.0),
    "nylon": Material(name="Nylon", E_MPa=1500.0, nu=0.40, density_kg_m3=1130.0, yield_MPa=45.0),
    "asa": Material(name="ASA", E_MPa=2000.0, nu=0.35, density_kg_m3=1070.0, yield_MPa=40.0),
}


def material_from_name(name: str | None) -> tuple[Material, str]:
    """Return (material, note). Unknown names fall back to PLA with an explicit note."""
    key = (name or "PLA").strip().lower().replace("-", "").replace(" ", "")
    if key in PRESETS:
        m = PRESETS[key].model_copy()
        m.confidence = "preset"
        return m, ""
    m = PRESETS["pla"].model_copy()
    m.confidence = "assumed"
    return m, f"material '{name}' unknown; using PLA properties"

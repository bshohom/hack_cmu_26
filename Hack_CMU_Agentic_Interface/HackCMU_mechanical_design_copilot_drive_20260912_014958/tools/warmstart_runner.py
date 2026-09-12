"""Execute an LLM-written warm-start script in a subprocess and write the candidate triple.

Standalone (no project imports) so it can run under any interpreter:

    python tools/warmstart_runner.py <script.py> <out_dir> <name> [envelope.json]

The script must define `PARAMS: dict`, `build(params) -> trimesh.Trimesh`, `DIMENSIONS: dict`,
`REGIONS: dict` and optionally `NOTES: list[str]`. Prints one JSON object on stdout.
The generated code runs with the caller's privileges; the prompt forbids I/O and the
subprocess has a timeout, but this is not a security sandbox.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import trimesh

N_PARTICLES = 75_000
MAX_FACES = 600_000
TRIMESH_VERSION = getattr(trimesh, "__version__", "unknown")


def cleanup_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Approved cleanup for the installed trimesh (no remove_duplicate_faces)."""
    if mesh is None:
        return mesh
    out = mesh.copy()
    out.merge_vertices()
    out.update_faces(out.unique_faces())
    out.remove_unreferenced_vertices()
    out.process(validate=True)
    return out


def union_all(parts) -> trimesh.Trimesh:
    """Deterministic solid assembly. concatenate() is not a valid substitute."""
    meshes = []
    for part in parts or []:
        if part is None:
            continue
        if not isinstance(part, trimesh.Trimesh):
            raise TypeError(f"union_all() expects trimesh.Trimesh parts, got {type(part).__name__}")
        if part.is_empty or len(part.faces) == 0:
            continue
        meshes.append(part)
    if not meshes:
        raise ValueError("union_all() received no meshes")
    if len(meshes) == 1:
        return meshes[0].copy()
    try:
        result = trimesh.boolean.union(meshes, engine="manifold", check_volume=False)
    except Exception as exc:  # noqa: BLE001 — reported to the generator
        raise RuntimeError(
            "union_all() failed with the installed manifold engine "
            f"({type(exc).__name__}: {exc}). Overlap adjoining primitives by >= 1 mm; "
            "coincident faces are not enough. Do not use trimesh.util.concatenate."
        ) from exc
    if result is None:
        raise RuntimeError("union_all() returned None. Overlap adjoining primitives by >= 1 mm.")
    if isinstance(result, (list, tuple)):
        raise RuntimeError(
            f"union_all() produced {len(result)} separate solids. "
            "Primitives must volumetrically overlap by >= 1 mm."
        )
    return result


def _install_trimesh_compat() -> None:
    """Map removed trimesh 3.x names onto 5.x equivalents for generated scripts."""
    if not hasattr(trimesh.Trimesh, "remove_duplicate_faces"):
        def remove_duplicate_faces(self) -> None:
            self.update_faces(self.unique_faces())
            self.remove_unreferenced_vertices()

        trimesh.Trimesh.remove_duplicate_faces = remove_duplicate_faces  # type: ignore[attr-defined]


def _load_script(path: Path):
    spec = importlib.util.spec_from_file_location("warmstart_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    _install_trimesh_compat()
    module.cleanup_mesh = cleanup_mesh
    module.union_all = union_all
    spec.loader.exec_module(module)
    return module


def as_box(region: dict | None) -> dict | None:
    """Accept {"min","max"} or {"name","box":{"min","max"}} (models emit both)."""
    if not isinstance(region, dict):
        return None
    if "min" in region and "max" in region:
        return region
    inner = region.get("box") or region.get("bounds")
    if isinstance(inner, dict) and "min" in inner and "max" in inner:
        return {**{k: v for k, v in region.items() if k != "box"}, **inner}
    return None


def _box(region: dict) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(region["min"], float), np.asarray(region["max"], float)


def _final_assembly_is_concatenate(script_src: str) -> bool:
    if "concatenate" not in (script_src or ""):
        return False
    returns = re.findall(r"^\s*return\s+(.+?)(?:#.*)?$", script_src, re.M)
    if returns and "concatenate" in returns[-1]:
        return True
    last_concat = script_src.rfind("concatenate")
    last_union = max(script_src.rfind("union_all("), script_src.rfind("boolean.union"))
    return last_concat > last_union


def validate(mesh: trimesh.Trimesh, regions: dict, envelope: dict, script_src: str = "") -> list[str]:
    problems: list[str] = []
    if mesh.is_empty or len(mesh.faces) == 0:
        return ["mesh has no faces"]
    if len(mesh.faces) > MAX_FACES:
        problems.append(f"too many faces ({len(mesh.faces)} > {MAX_FACES}); use fewer segments")
    if _final_assembly_is_concatenate(script_src):
        problems.append(
            "ASSEMBLY REJECTED: final assembly uses trimesh.util.concatenate. That glues "
            "triangle soups and cannot produce one solid. Overlap adjoining primitives by "
            ">= 1 mm and return union_all(parts)."
        )
    if not mesh.is_watertight:
        problems.append(
            "mesh is not watertight; overlap adjoining primitives by >= 1 mm and assemble with "
            "union_all(parts). Coincident faces and concatenate() are not enough."
        )
    parts = mesh.split(only_watertight=False)
    if len(parts) != 1:
        problems.append(f"mesh has {len(parts)} disconnected bodies; it must be one connected body")
        problems.extend(_component_diagnostics(parts))
    if mesh.is_watertight and mesh.volume <= 0:
        problems.append("mesh volume is not positive (inverted normals?)")
    lo, hi = mesh.bounds
    ext = hi - lo
    max_prot = envelope.get("max_protrusion_mm")
    if max_prot is not None and hi[0] > max_prot + 1e-6:
        problems.append(f"part protrudes to x={hi[0]:.1f} mm, beyond max_protrusion_mm={max_prot}")
    max_w = envelope.get("max_width_mm")
    if max_w is not None and ext[1] > max_w + 1e-6:
        problems.append(f"part width {ext[1]:.1f} mm exceeds max_width_mm={max_w}")
    max_h = envelope.get("max_height_mm")
    if max_h is not None and ext[2] > max_h + 1e-6:
        problems.append(f"part height {ext[2]:.1f} mm exceeds max_height_mm={max_h}")
    for key in ("load", "mounts"):
        if key not in regions:
            problems.append(f"REGIONS is missing '{key}'")
    load = as_box(regions.get("load"))
    if load is not None:
        lmin, lmax = _box(load)
        if np.any(lmax < lo) or np.any(lmin > hi):
            problems.append(
                f"REGIONS['load'] box {np.round(lmin,1).tolist()}..{np.round(lmax,1).tolist()} lies outside the "
                f"part bounds {np.round(lo,1).tolist()}..{np.round(hi,1).tolist()}. It must be a slab of YOUR "
                "material on the load-bearing face, not empty space or the payload volume."
            )
        else:
            problems += _region_touches_part(mesh, load, "REGIONS['load']")
    mounts = regions.get("mounts") or []
    if isinstance(mounts, list) and not mounts:
        problems.append("REGIONS['mounts'] must list at least one contact box")
    for i, raw in enumerate(mounts if isinstance(mounts, list) else []):
        m = as_box(raw)
        if m is not None:
            problems += _region_touches_part(mesh, m, f"REGIONS['mounts'][{i}] ({m.get('name', '')})")
    return problems


def _nearest_component_distance(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    try:
        pts, _ = trimesh.sample.sample_surface(a, 250)
        _, dist, _ = trimesh.proximity.closest_point(b, pts)
        return float(np.min(dist))
    except Exception:  # noqa: BLE001 — fall back to bounding-box gap
        alo, ahi = a.bounds
        blo, bhi = b.bounds
        gap = np.maximum(alo - bhi, blo - ahi)
        return float(np.linalg.norm(np.maximum(gap, 0.0)))


def _component_diagnostics(parts: list) -> list[str]:
    lines: list[str] = []
    for i, part in enumerate(parts):
        lo, hi = part.bounds
        lines.append(
            f"  component[{i}] bbox {np.round(lo, 2).tolist()} .. {np.round(hi, 2).tolist()}"
        )
    min_d = None
    pair = None
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            dist = _nearest_component_distance(parts[i], parts[j])
            if min_d is None or dist < min_d:
                min_d = dist
                pair = (i, j)
    if pair is not None and min_d is not None:
        if min_d <= 1e-3:
            lines.append(
                f"  nearest gap: component[{pair[0]}] to component[{pair[1]}] = 0 mm "
                "(they touch or nearly touch but were not boolean-unioned). "
                "Overlap those two by >= 1 mm and return union_all(parts)."
            )
        else:
            lines.append(
                f"  nearest gap: component[{pair[0]}] to component[{pair[1]}] = {min_d:.2f} mm. "
                "Move them so they overlap by >= 1 mm and return union_all(parts)."
            )
    return lines


def _region_touches_part(mesh: trimesh.Trimesh, region: dict, label: str, tol_mm: float = 3.0, min_frac: float = 0.25) -> list[str]:
    """A load/mount box must actually sit on the part: >= min_frac of a sample grid
    inside the box lies within tol_mm of the mesh surface."""
    lo, hi = _box(region)
    axes = [np.linspace(lo[i], hi[i], 6) for i in range(3)]
    pts = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    try:
        _, dist, _ = trimesh.proximity.closest_point(mesh, pts)
    except Exception:  # noqa: BLE001 — proximity needs rtree; skip the check without it
        return []
    frac = float(np.mean(dist <= tol_mm))
    if frac < min_frac:
        cen = np.round((lo + hi) / 2, 1).tolist()
        return [
            f"{label} box centred at {cen} does not touch the part (only {frac:.0%} of its volume is within "
            f"{tol_mm} mm of the surface); place it on the actual face/floor where that load or contact occurs"
        ]
    return []


def write_triple(mesh: trimesh.Trimesh, out_dir: Path, name: str, dims: dict, notes: list, regions: dict, params: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stl = out_dir / f"{name}.stl"
    mesh.export(stl)
    pts, _ = trimesh.sample.sample_surface(mesh, N_PARTICLES)
    particles = out_dir / f"{name}_particles.obj"
    with particles.open("w") as f:
        f.write(f"# {name} surface point cloud; units mm\n")
        for p in pts:
            f.write(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
    lo, hi = mesh.bounds
    lines = [f"{name} — generated warm start", "Units: mm", "", "DESIGN DIMENSIONS"]
    lines += [f"- {k}: {v}" for k, v in dims.items()]
    if notes:
        lines += ["", "NOTES"] + [f"- {n}" for n in notes]
    lines += [
        "",
        "MESH",
        f"- Vertices: {len(mesh.vertices)}",
        f"- Faces: {len(mesh.faces)}",
        f"- Watertight: {mesh.is_watertight}",
        f"- Connected components: {len(mesh.split(only_watertight=False))}",
        f"- Bounding box min: {np.round(lo, 2).tolist()}",
        f"- Bounding box max: {np.round(hi, 2).tolist()}",
    ]
    dims_path = out_dir / f"{name}_dimensions.txt"
    dims_path.write_text("\n".join(lines) + "\n")
    regions_path = out_dir / f"{name}_regions.json"
    regions_path.write_text(json.dumps({"frame": "desk_edge_frame", "units": "mm", "params": params, "regions": regions}, indent=2))
    return {
        "mesh_path": str(stl),
        "particle_path": str(particles),
        "dimensions_path": str(dims_path),
        "regions_path": str(regions_path),
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "connected_components": int(len(mesh.split(only_watertight=False))),
        "volume_mm3": float(mesh.volume) if mesh.is_watertight else None,
        "bbox_min_mm": np.round(lo, 3).tolist(),
        "bbox_max_mm": np.round(hi, 3).tolist(),
    }


def main(argv: list[str]) -> int:
    script, out_dir, name = Path(argv[1]), Path(argv[2]), argv[3]
    envelope = json.loads(Path(argv[4]).read_text()) if len(argv) > 4 else {}
    result: dict = {"ok": False, "name": name}
    try:
        module = _load_script(script)
        params = dict(getattr(module, "PARAMS", {}))
        mesh = module.build(params)
        if isinstance(mesh, (list, tuple)):
            mesh = union_all(mesh)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"build() must return trimesh.Trimesh or a list of parts, got {type(mesh).__name__}")
        mesh = cleanup_mesh(mesh)
        regions = dict(getattr(module, "REGIONS", {}))
        dims = dict(getattr(module, "DIMENSIONS", {}))
        notes = list(getattr(module, "NOTES", []))
        problems = validate(mesh, regions, envelope, script_src=script.read_text())
        result["problems"] = problems
        result.update(write_triple(mesh, out_dir, name, dims, notes, regions, params))
        result["ok"] = not problems
    except Exception:  # noqa: BLE001 — everything is reported to the caller
        result["error"] = traceback.format_exc(limit=6)
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))

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
import sys
import traceback
from pathlib import Path

import numpy as np
import trimesh

N_PARTICLES = 75_000
MAX_FACES = 600_000


def _load_script(path: Path):
    spec = importlib.util.spec_from_file_location("warmstart_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _box(region: dict) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(region["min"], float), np.asarray(region["max"], float)


def validate(mesh: trimesh.Trimesh, regions: dict, envelope: dict) -> list[str]:
    problems: list[str] = []
    if mesh.is_empty or len(mesh.faces) == 0:
        return ["mesh has no faces"]
    if len(mesh.faces) > MAX_FACES:
        problems.append(f"too many faces ({len(mesh.faces)} > {MAX_FACES}); use fewer segments")
    if not mesh.is_watertight:
        problems.append("mesh is not watertight; union all bodies with trimesh.boolean.union(..., engine='manifold') and avoid coincident faces")
    parts = mesh.split(only_watertight=False)
    if len(parts) != 1:
        problems.append(f"mesh has {len(parts)} disconnected bodies; it must be one connected body")
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
    load = regions.get("load")
    if isinstance(load, dict) and "min" in load and "max" in load:
        lmin, lmax = _box(load)
        if np.any(lmax < lo) or np.any(lmin > hi):
            problems.append("REGIONS['load'] box does not overlap the part bounding box")
        else:
            problems += _region_touches_part(mesh, load, "REGIONS['load']")
    mounts = regions.get("mounts") or []
    if isinstance(mounts, list) and not mounts:
        problems.append("REGIONS['mounts'] must list at least one contact box")
    for i, m in enumerate(mounts if isinstance(mounts, list) else []):
        if isinstance(m, dict) and "min" in m and "max" in m:
            problems += _region_touches_part(mesh, m, f"REGIONS['mounts'][{i}] ({m.get('name', '')})")
    return problems


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
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"build() must return trimesh.Trimesh, got {type(mesh).__name__}")
        mesh = mesh.copy()
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()
        regions = dict(getattr(module, "REGIONS", {}))
        dims = dict(getattr(module, "DIMENSIONS", {}))
        notes = list(getattr(module, "NOTES", []))
        problems = validate(mesh, regions, envelope)
        result["problems"] = problems
        result.update(write_triple(mesh, out_dir, name, dims, notes, regions, params))
        result["ok"] = not problems
    except Exception:  # noqa: BLE001 — everything is reported to the caller
        result["error"] = traceback.format_exc(limit=6)
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))

# to_agent — agent-facing 3D topology optimization on torch-fem

Turns a plain-data problem description (regions, supports, loads, material, volume fraction)
into a hex-grid SIMP topology optimization on [torch-fem](https://github.com/meyer-nils/torch-fem)
and writes a printable STL plus diagnostics. The agent layer only has to emit a `TOProblem` YAML.

## Setup (fresh conda env, GPU)

```bash
conda create -n hack_cmu_26 python=3.12 -y
conda activate hack_cmu_26
pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 5080 / sm_120
pip install -e ".[dev]"
python scripts/smoke_cantilever.py        # torch-fem's own example: numeric stack sanity check
python scripts/calibrate_cost.py          # times solves on this machine -> data/calibration.json
pytest -q
```

## Demo: desk-clamp cupholder

```bash
to-agent make-cupholder-problem            # dims + scan -> data/cupholder_problem.yaml
to-agent validate data/cupholder_problem.yaml
to-agent estimate data/cupholder_problem.yaml        # [ok|slow|too_big] + seconds/memory
to-agent run data/cupholder_problem.yaml --out out   # rho.vti, design.stl, history.png, render.png, result.json
to-agent run data/cupholder_problem.yaml --out out_fine --elem 3 --volfrac 0.15 --iters 60   # overrides
```

`python -m to_agent ...` works without the console script.

## Problem schema (`TOProblem`)

```yaml
units: {length: mm, force: N}
material: {name: PLA, E_MPa: 2300, nu: 0.35}
design_domain: {type: box, min: [x0, y0, z0], max: [x1, y1, z1]}   # or omit -> padded warm-start bounds
warm_start: [ {type: near_points, path: scan.obj, tol: 4.0} ]      # or inside_mesh / near_mesh
preserve:   [ ...regions held solid (attachment interfaces)... ]
void:       [ ...regions kept empty (cavities, the desk, clearance)... ]
supports:   [ {id: jaw, region: {...}, fixed_dofs: [x, y, z]} ]
load_cases: [ {id: drink, region: {...}, force_N: [0, 0, -10], weight: 1.0} ]   # total force over region nodes
safety_factor: 2.5          # multiplies every load
volume_fraction: 0.35       # of the design elements
target_element_size: 4.0    # mm; grid = design_domain / element size
filter_radius: 6.0          # default 1.5 * element size
max_iters: 40
```

Region primitives: `box`, `cylinder` (annulus via `r_min`, extent via `along`), `sphere`,
`halfspace`, `near_points` (point cloud ± tol), `inside_mesh` (watertight STL/OBJ),
`near_mesh` (surface ± tol), and `union` / `intersection` / `difference`.
Element classification priority: void > preserve > design. Loads/supports select grid nodes;
a region that selects no nodes raises an agent-readable error.

## Layout

```
to_agent/contracts.py      pydantic TOProblem + region primitives, YAML load/save
to_agent/regions.py        contains()/bounds() for every primitive
to_agent/ingest/           point clouds, meshes, dimension files
to_agent/meshing/          cube_hexa grid + element/node masks
to_agent/solver/simp.py    SIMP (OC update, sparse filter, multi-load-case) on torch-fem
to_agent/cost.py           time/memory estimate, calibrated by scripts/calibrate_cost.py
to_agent/postprocess/      VTI + STL export, history plot, render
to_agent/demo/cupholder.py the only part-specific code (dims + scan -> TOProblem)
```

## Data notes

- `cupholder_surface_particles.obj` is a 50k-point surface sample (no faces); its clamp levels
  (plate underside, hook top, spine) are measured from the scan by `demo/cupholder.py` and
  cross-checked against `cupholder_dimensions.txt`.
- Loads in `data/cupholder_problem.yaml` are placeholders (`confidence: assumed`).
- GPU: the model is built on CUDA when torch-fem supports it, otherwise the linear solver runs
  on CUDA with the model on CPU (`solver_mode` in result.json says which).

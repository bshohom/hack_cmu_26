# Integration notes (Shohom, 2026-09-12)

The orchestration/contract layer now drives real tools. Everything below is backward
compatible: all pre-existing tests pass unchanged, the mock paths are the defaults, and
every new schema field has a default.

## Run

```bash
conda activate hack_cmu_26            # torch + torch-fem + to_agent (editable) + streamlit/plotly
cd Hack_CMU_Agentic_Interface/HackCMU_mechanical_design_copilot_drive_20260912_014958
streamlit run app.py --server.fileWatcherType none     # .streamlit/config.toml also sets this
```

`.env` here holds `GROK_API_KEY` (git-ignored). `pytest -q` → 68 tests.

## What is live now

| Stage | Before | Now |
|---|---|---|
| Warm-start geometry | none (external files) | **Geometry source "Generated Warm Start (Grok)"**: `tools/warmstart.py` asks Grok for a constrained trimesh script, runs it in a subprocess (`tools/warmstart_runner.py`), validates (watertight, one body, inside the envelope, load/mount boxes touch the part), retries ≤3 with the validator's feedback, writes `<name>.stl + _particles.obj + _dimensions.txt + _regions.json` under `generated/`. Called from `Orchestrator._handle_geometry` via `warm_start_generator`. |
| Candidate import | cupholder OBJ only | `imported_candidate.CANDIDATES` registry (cupholder, desk_bag_hook, stapler_shelf — triples at the repo root), `load_candidate(name)`, trimesh stats for STL, bullet/`Z=` tolerant dimension parsing, fit-gate mapping for clamp parts. Sidebar candidate selector + "Load Desk Hook / Stapler Shelf Case (live TO)" presets. |
| Topology optimization | mock | **Topology = Live**: `tools/topology.py` → `to_agent.integration.agentic.run_topology` (SIMP on torch-fem, GPU). `TopologyInput.candidate/solver_options/desk_thickness_mm`; `TopologyOutput.artifacts/notes/iterations/wall_time_s/converged/post_check`. Any failure → mock with the reason in `notes`. |
| CAD | mock filename | `tools/cad.py` passes the optimized STL through (`is_mock=False`) when it exists. |
| Verification | always UNVERIFIED | still UNVERIFIED by design; notes now carry the post-TO linear FE check (max displacement, max von Mises, factor of safety at nominal load). |
| Registration | mock fixture | sidebar `target.json` path → `to_agent.ingest.surfcap.target_to_measurements` (metres → mm, confidence from `scale.reliable`/`rms_mm`/views). Confidence ≥ 0.7 pre-fills the desk-thickness question for confirmation; lower → the user is asked; a disagreement is shown, never silently resolved. A trusted measurement becomes a live `RegistrationOutput`. |

## Files touched

`schemas.py` (new optional fields + `TopologySolverOptions`), `orchestrator.py` (candidate/options into
`TopologyInput`, `warm_start_generator`, trace + verification notes), `tools/topology.py`, `tools/cad.py`,
`tools/warmstart.py` (new), `tools/warmstart_runner.py` (new), `imported_candidate.py` (registry, loader,
parsers), `geometry_sources.py` (`GEOM_GENERATED`, `topology_live`, candidate name), `ui_viz.py` (trimesh
mesh loader, design-over-candidate figure), `app.py` (sidebar controls, presets, result block, registration
prefill), `scripts/grok_warmstart_probe.py` (new), `test_topology_tool.py` (new), `.streamlit/config.toml`,
`README_INTEGRATION.md`.

## Frames and units

Everything is mm / N. The **candidate mesh's own frame is the optimization frame**. Grok-generated
parts use `desk_edge_frame` (desk underside z=0, desk top z=desk_thickness, desk front edge x=0 with
the desk at x<0, part toward +x, +Z up), so the adapter builds their problem generically from
`_regions.json` (load box, mount boxes, keep-outs) plus a desk-slab void. Imported demo parts use
per-task templates in `to_agent/demo/`. surfcap's frame (card-centred, metres) is only used for
scalars (desk thickness, mount normal), never for mesh registration.

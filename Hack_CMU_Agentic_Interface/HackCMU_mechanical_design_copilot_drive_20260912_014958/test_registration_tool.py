"""Fast contract tests for the external surfcap registration skill."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.registration import (
    CaptureError,
    UploadedPhoto,
    build_command,
    load_registration_output,
    load_sample_registration,
    run_registration,
    run_sample_registration,
    stage_uploads,
    validate_capture,
)


def _capture(folder: Path, count: int = 8) -> Path:
    folder.mkdir(parents=True)
    for i in range(count):
        (folder / f"photo_{i:02d}.jpg").write_bytes(b"image")
    return folder


def _target() -> dict:
    return {
        "frame": {
            "units": "m",
            "up": [0, 0, 1],
            "origin_desc": "card centre on mount plane",
        },
        "scale": {"reliable": True, "rms_mm": 1.0, "n_views_with_ref": 8},
        "surfaces": [
            {
                "role": "top",
                "normal": [0, 0, 1],
                "centroid": [0, 0, 0],
                "extent_m": [0.6, 0.4],
                "polygon_3d": [
                    [-0.3, -0.2, 0],
                    [0.3, -0.2, 0],
                    [0.3, 0.2, 0],
                    [-0.3, 0.2, 0],
                ],
            },
            {
                "role": "front",
                "normal": [0, -1, 0],
                "centroid": [0, -0.2, -0.01],
                "extent_m": [0.6, 0.02],
            },
        ],
        "cloud": {"postprocess": {}},
        "warnings": [],
    }


def test_capture_validation_enforces_bounds(tmp_path: Path) -> None:
    too_small = _capture(tmp_path / "small", 7)
    with pytest.raises(CaptureError, match="at least 8"):
        validate_capture(too_small)

    usable = _capture(tmp_path / "usable", 8)
    images, warnings = validate_capture(usable)
    assert len(images) == 8
    assert any("recommended" in warning for warning in warnings)


def test_stage_uploads_sanitizes_names(tmp_path: Path) -> None:
    photos = [
        UploadedPhoto(name=f"phone image {i}.JPG", data=b"image")
        for i in range(8)
    ]
    folder, _ = stage_uploads(photos, tmp_path / "capture")
    assert len(list(folder.glob("*.jpg"))) == 8
    assert all(" " not in path.name for path in folder.iterdir())


def test_command_uses_isolated_cli_and_sfm(tmp_path: Path) -> None:
    command = build_command(
        tmp_path / "capture",
        tmp_path / "out",
        python="/opt/hack/bin/python",
        table_prompt="desk",
    )
    assert command[:3] == ["/opt/hack/bin/python", "-m", "surfcap"]
    assert command[command.index("--recon-mode") + 1] == "sfm_tsdf"
    assert command[command.index("--table-prompt") + 1] == "desk"


def test_run_registration_consumes_target_contract(tmp_path: Path) -> None:
    capture = _capture(tmp_path / "capture")
    root = tmp_path / "HackCMU"
    (root / "surfcap").mkdir(parents=True)
    (root / "surfcap" / "__main__.py").write_text("")

    def fake_run(command, **_kwargs):
        out_dir = Path(command[command.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "target.json").write_text(json.dumps(_target()))
        (out_dir / "target.glb").write_bytes(b"glb")
        return subprocess.CompletedProcess(command, 0, stdout="complete", stderr="")

    with patch.dict(os.environ, {"SURFCAP_MODE": "live"}), patch(
        "tools.registration.subprocess.run", side_effect=fake_run
    ):
        result = run_registration(
            capture,
            out_root=tmp_path / "runs",
            surfcap_root=root,
            python="/opt/hack/bin/python",
        )

    assert result.ok
    assert result.measurements["desk_thickness_mm"] == 20.0
    assert result.measurements["prefill"] is True
    assert result.measurements["mount_polygon_mm"][0] == [-300.0, -200.0, 0.0]
    assert result.artifacts["viewer"].endswith("target.glb")


def test_run_registration_rejects_missing_artifact(tmp_path: Path) -> None:
    capture = _capture(tmp_path / "capture")
    root = tmp_path / "HackCMU"
    (root / "surfcap").mkdir(parents=True)
    (root / "surfcap" / "__main__.py").write_text("")
    completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")

    with patch.dict(os.environ, {"SURFCAP_MODE": "live"}), patch(
        "tools.registration.subprocess.run", return_value=completed
    ):
        result = run_registration(
            capture,
            out_root=tmp_path / "runs",
            surfcap_root=root,
        )

    assert not result.ok
    assert "did not write" in result.message
    assert not result.measurements


def test_load_existing_hybrid_output_as_callable_skill(tmp_path: Path) -> None:
    import trimesh

    output = tmp_path / "HackCMU" / "out" / "table_a"
    output.mkdir(parents=True)
    (output / "target.json").write_text(json.dumps(_target()))
    trimesh.creation.box(extents=[0.6, 0.4, 0.02]).export(
        output / "target_mesh_hybrid.ply"
    )
    (output / "target_mesh_hybrid.glb").write_bytes(b"glb")

    direct = load_registration_output(output / "target_mesh_hybrid.ply")
    sample = load_sample_registration("table_a", surfcap_root=tmp_path / "HackCMU")

    for result in (direct, sample):
        assert result.ok
        assert result.measurements["desk_thickness_mm"] == 20.0
        assert result.measurements["prefill"] is True
        assert result.measurements["mesh_extent_mm"] == [600.0, 400.0, 20.0]
        assert result.artifacts["hybrid_mesh"].endswith("target_mesh_hybrid.ply")


def test_bundled_sample_invokes_same_registration_skill(tmp_path: Path) -> None:
    capture = _capture(tmp_path / "examples" / "surfcap" / "table_a")
    expected = object()
    with patch("tools.registration.PROJECT_ROOT", tmp_path), patch(
        "tools.registration.run_registration", return_value=expected
    ) as mocked:
        result = run_sample_registration(
            "table_a", out_root=tmp_path / "out", python="/opt/surfcap/bin/python"
        )

    assert result is expected
    mocked.assert_called_once_with(
        capture,
        out_root=tmp_path / "out",
        python="/opt/surfcap/bin/python",
    )

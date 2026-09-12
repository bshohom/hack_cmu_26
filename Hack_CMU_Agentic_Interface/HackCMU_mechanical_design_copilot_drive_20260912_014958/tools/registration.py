"""Callable grounding-surface registration skill backed by the in-repo surfcap package.

The heavy CV stack stays isolated from Streamlit: this module stages a capture, invokes
``python -m surfcap`` in a subprocess, and consumes the resulting target.json through
to_agent's existing trust/provenance adapter.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic"}
MIN_IMAGES = 8
RECOMMENDED_IMAGES = 14
MAX_IMAGES = 18
RUN_TIMEOUT_S = 15 * 60

INTERFACE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_ROOT = INTERFACE_ROOT / "generated" / "registration"

CAPTURE_GUIDANCE = (
    "Place a matte, contrasting credit/debit/ID card flat on the mounting surface, "
    "10–20 cm from the edge of interest.",
    "Do not move the card, surface, or nearby objects during the capture.",
    "Take 14–18 overlapping photos in a 120–180° arc, using the phone's 1× lens.",
    "Use two heights: mostly downward views plus 6–8 low views showing the edge/underside.",
    "Include two close card views and keep the card visible in at least eight photos.",
    "Avoid blur, zoom, portrait mode, changing light, and reflective cards.",
)


class CaptureError(ValueError):
    """The supplied capture cannot be submitted to surfcap."""


@dataclass(frozen=True)
class UploadedPhoto:
    name: str
    data: bytes


@dataclass
class RegistrationRunResult:
    ok: bool
    message: str
    capture_dir: str = ""
    out_dir: str = ""
    target_json: str = ""
    measurements: dict = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    log: str = ""
    elapsed_s: float = 0.0
    command: list[str] = field(default_factory=list)


def _default_surfcap_root() -> Path:
    standalone = PROJECT_ROOT
    nested = PROJECT_ROOT / "HackCMU"
    sibling = PROJECT_ROOT.parent / "HackCMU"
    if (standalone / "surfcap" / "__main__.py").is_file():
        return standalone
    return nested if nested.is_dir() else sibling


def _default_surfcap_python() -> Path:
    configured = os.environ.get("SURFCAP_PYTHON")
    if configured:
        return Path(configured).expanduser()
    return Path(sys.executable)


def _safe_name(name: str, index: int) -> str:
    base = Path(name).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(base).stem).strip("._")
    suffix = Path(base).suffix.lower()
    return f"{index:02d}_{stem or 'photo'}{suffix}"


def _capture_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise CaptureError(f"capture folder does not exist: {folder}")
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def validate_capture(folder: str | Path) -> tuple[list[Path], list[str]]:
    """Validate a folder and return its images plus non-fatal capture warnings."""
    folder = Path(folder).expanduser()
    images = _capture_images(folder)
    if len(images) < MIN_IMAGES:
        raise CaptureError(
            f"capture has {len(images)} supported images; at least {MIN_IMAGES} are required"
        )
    if len(images) > MAX_IMAGES:
        raise CaptureError(
            f"capture has {len(images)} supported images; select at most {MAX_IMAGES}"
        )
    warnings: list[str] = []
    if len(images) < RECOMMENDED_IMAGES:
        warnings.append(
            f"{len(images)} photos is usable but below the recommended "
            f"{RECOMMENDED_IMAGES}–{MAX_IMAGES}"
        )
    return images, warnings


def stage_uploads(
    photos: Iterable[UploadedPhoto],
    capture_dir: str | Path,
) -> tuple[Path, list[str]]:
    """Write uploaded image bytes into a private run directory and validate the set."""
    photos = list(photos)
    if not MIN_IMAGES <= len(photos) <= MAX_IMAGES:
        raise CaptureError(
            f"select {MIN_IMAGES}–{MAX_IMAGES} photos; received {len(photos)}"
        )
    capture_dir = Path(capture_dir)
    capture_dir.mkdir(parents=True, exist_ok=False)
    seen: set[str] = set()
    for index, photo in enumerate(photos, start=1):
        suffix = Path(photo.name).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise CaptureError(
                f"unsupported image {photo.name!r}; use JPG, PNG, or HEIC"
            )
        name = _safe_name(photo.name, index)
        if name in seen:
            raise CaptureError(f"duplicate staged image name: {name}")
        seen.add(name)
        if not photo.data:
            raise CaptureError(f"uploaded image {photo.name!r} is empty")
        (capture_dir / name).write_bytes(photo.data)
    _, warnings = validate_capture(capture_dir)
    return capture_dir, warnings


def build_command(
    capture_dir: str | Path,
    out_dir: str | Path,
    *,
    python: str | Path | None = None,
    table_prompt: str = "table",
) -> list[str]:
    """Build the stable CLI invocation.  Keep sfm_tsdf explicit."""
    return [
        str(Path(python) if python else _default_surfcap_python()),
        "-m",
        "surfcap",
        str(Path(capture_dir).resolve()),
        "--out",
        str(Path(out_dir).resolve()),
        "--recon-mode",
        "sfm_tsdf",
        "--res",
        "512",
        "--n-max",
        "10",
        "--table-prompt",
        table_prompt.strip() or "table",
    ]


def _artifact_paths(out_dir: Path) -> dict[str, str]:
    names = {
        "target_json": "target.json",
        "point_cloud": "target.ply",
        "viewer": "target.glb",
        "planar_mesh": "target_mesh.ply",
        "planar_mesh_viewer": "target_mesh.glb",
        "hybrid_mesh": "target_mesh_hybrid.ply",
        "hybrid_mesh_viewer": "target_mesh_hybrid.glb",
        "top_view": "debug/world_top.png",
        "side_view": "debug/world_side.png",
    }
    return {
        key: str(path)
        for key, rel in names.items()
        if (path := out_dir / rel).is_file() and path.stat().st_size > 0
    }


def load_registration_output(path: str | Path) -> RegistrationRunResult:
    """Load an existing read-only surfcap output directory, target contract, or mesh."""
    started = time.monotonic()
    supplied = Path(path).expanduser().resolve()
    out_dir = supplied if supplied.is_dir() else supplied.parent
    target_path = out_dir / "target.json"
    measurement_path = supplied
    if supplied.is_dir():
        measurement_path = (
            out_dir / "target_mesh_hybrid.ply"
            if (out_dir / "target_mesh_hybrid.ply").is_file()
            else target_path
        )
    if not measurement_path.is_file():
        return RegistrationRunResult(
            ok=False,
            message=f"registration output does not exist: {measurement_path}",
            out_dir=str(out_dir),
        )
    try:
        from to_agent.ingest.surfcap import measurements_from_path

        measurements = measurements_from_path(measurement_path)
    except Exception as exc:  # noqa: BLE001 — malformed producer output is reported
        return RegistrationRunResult(
            ok=False,
            message=f"invalid registration output: {type(exc).__name__}: {exc}",
            out_dir=str(out_dir),
            elapsed_s=round(time.monotonic() - started, 2),
        )
    return RegistrationRunResult(
        ok=True,
        message="existing grounding-surface reconstruction loaded; review and confirm measurements",
        out_dir=str(out_dir),
        target_json=str(target_path if target_path.is_file() else measurement_path),
        measurements=measurements,
        artifacts=_artifact_paths(out_dir),
        warnings=list(dict.fromkeys(str(v) for v in measurements.get("notes", []))),
        elapsed_s=round(time.monotonic() - started, 2),
    )


def load_sample_registration(
    name: str = "table_a", *, surfcap_root: str | Path | None = None
) -> RegistrationRunResult:
    """Load a generated sample output when one already exists under the runtime root."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        return RegistrationRunResult(ok=False, message=f"invalid sample name: {name!r}")
    root = Path(
        surfcap_root or os.environ.get("SURFCAP_ROOT") or _default_surfcap_root()
    ).expanduser().resolve()
    return load_registration_output(root / "out" / name)


def run_registration(
    capture_folder: str | Path,
    *,
    out_root: str | Path = DEFAULT_OUT_ROOT,
    surfcap_root: str | Path | None = None,
    python: str | Path | None = None,
    table_prompt: str = "table",
    timeout_s: int = RUN_TIMEOUT_S,
    log: Optional[Callable[[str], None]] = None,
) -> RegistrationRunResult:
    """Run surfcap and return measurements without fabricating a fallback result."""
    started = time.monotonic()
    try:
        capture_dir = Path(capture_folder).expanduser().resolve()
        _, warnings = validate_capture(capture_dir)
    except CaptureError as exc:
        return RegistrationRunResult(ok=False, message=str(exc))

    if os.environ.get("SURFCAP_MODE", "live").strip().lower() == "off":
        return RegistrationRunResult(
            ok=False,
            message="SURFCAP_MODE=off: grounding-surface reconstruction is disabled",
            capture_dir=str(capture_dir),
            warnings=warnings,
        )

    root = Path(
        surfcap_root or os.environ.get("SURFCAP_ROOT") or _default_surfcap_root()
    ).expanduser().resolve()
    if not (root / "surfcap" / "__main__.py").is_file():
        return RegistrationRunResult(
            ok=False,
            message=(
                f"surfcap checkout not found at {root}; set SURFCAP_ROOT to the "
                "repository containing the surfcap package"
            ),
            capture_dir=str(capture_dir),
            warnings=warnings,
        )

    out_root = Path(out_root)
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000:06d}"
    out_dir = out_root / f"surface_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=False)
    command = build_command(
        capture_dir, out_dir, python=python, table_prompt=table_prompt
    )
    if log:
        log(f"surfcap: reconstructing {len(_capture_images(capture_dir))} photos")
    try:
        completed = subprocess.run(
            command,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return RegistrationRunResult(
            ok=False,
            message=f"surfcap could not complete: {type(exc).__name__}: {exc}",
            capture_dir=str(capture_dir),
            out_dir=str(out_dir),
            warnings=warnings,
            elapsed_s=round(time.monotonic() - started, 2),
            command=command,
        )

    combined_log = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    target_path = out_dir / "target.json"
    if completed.returncode != 0:
        return RegistrationRunResult(
            ok=False,
            message=f"surfcap exited with status {completed.returncode}",
            capture_dir=str(capture_dir),
            out_dir=str(out_dir),
            warnings=warnings,
            log=combined_log[-8000:],
            elapsed_s=round(time.monotonic() - started, 2),
            command=command,
        )
    if not (target_path.is_file() and target_path.stat().st_size > 0):
        return RegistrationRunResult(
            ok=False,
            message=f"surfcap reported success but did not write {target_path}",
            capture_dir=str(capture_dir),
            out_dir=str(out_dir),
            warnings=warnings,
            log=combined_log[-8000:],
            elapsed_s=round(time.monotonic() - started, 2),
            command=command,
        )

    try:
        from to_agent.ingest.surfcap import measurements_from_path

        measurements = measurements_from_path(target_path)
    except Exception as exc:  # noqa: BLE001 - malformed producer output is a tool failure
        return RegistrationRunResult(
            ok=False,
            message=f"invalid surfcap target.json: {type(exc).__name__}: {exc}",
            capture_dir=str(capture_dir),
            out_dir=str(out_dir),
            target_json=str(target_path),
            warnings=warnings,
            log=combined_log[-8000:],
            elapsed_s=round(time.monotonic() - started, 2),
            command=command,
        )

    warnings.extend(str(v) for v in measurements.get("notes", []))
    return RegistrationRunResult(
        ok=True,
        message="grounding surface reconstructed; review and confirm measurements",
        capture_dir=str(capture_dir),
        out_dir=str(out_dir),
        target_json=str(target_path),
        measurements=measurements,
        artifacts=_artifact_paths(out_dir),
        warnings=list(dict.fromkeys(warnings)),
        log=combined_log[-8000:],
        elapsed_s=round(time.monotonic() - started, 2),
        command=command,
    )


def run_sample_registration(
    name: str = "table_a",
    *,
    out_root: str | Path = DEFAULT_OUT_ROOT,
    python: str | Path | None = None,
) -> RegistrationRunResult:
    """Run the callable skill against a bundled capture under examples/surfcap."""
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        return RegistrationRunResult(ok=False, message=f"invalid sample name: {name!r}")
    capture = PROJECT_ROOT / "examples" / "surfcap" / name
    if not capture.is_dir():
        return RegistrationRunResult(
            ok=False, message=f"bundled surfcap capture does not exist: {capture}"
        )
    return run_registration(capture, out_root=out_root, python=python)

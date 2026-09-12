import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "examples" / "surfcap" / "table_a"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic"}


def test_surfcap_cli_is_packaged_in_this_repository():
    assert (ROOT / "surfcap" / "__init__.py").is_file()
    assert (ROOT / "surfcap" / "__main__.py").is_file()
    completed = subprocess.run(
        [sys.executable, "-m", "surfcap", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--recon-mode" in completed.stdout


def test_table_a_capture_is_bundled_for_reconstruction():
    images = [
        path
        for path in SAMPLE.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    assert 8 <= len(images) <= 18
    assert (SAMPLE / "notes.txt").is_file()

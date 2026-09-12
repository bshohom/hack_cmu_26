"""Device selection. Import this module before torch anywhere in the package."""

from __future__ import annotations

import os

# An empty CUDA_VISIBLE_DEVICES hides every GPU; treat it as unset.
if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
    del os.environ["CUDA_VISIBLE_DEVICES"]

import torch  # noqa: E402

# torch-fem's reference examples run in float64; keep the whole package consistent.
torch.set_default_dtype(torch.float64)


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return dev


def device_memory_bytes(device: torch.device) -> int:
    """Total memory of the device (GPU VRAM or host RAM)."""
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).total_memory
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return f"cuda ({props.name}, {props.total_memory / 1e9:.1f} GB, sm_{props.major}{props.minor})"
    return f"cpu ({device_memory_bytes(device) / 1e9:.1f} GB RAM)"

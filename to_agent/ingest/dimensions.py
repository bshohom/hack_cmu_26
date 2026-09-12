"""Parse simple `Key: value` dimension files (as produced alongside warm-start meshes)."""

from __future__ import annotations

import re
from pathlib import Path

_RANGE = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?)\s*[–-]\s*([-+]?\d+(?:\.\d+)?)\s*$")
# "5 kg static", "Z=100.0", "About 86mm" -> the leading number
_LEADING_NUMBER = re.compile(r"^\s*(?:[A-Za-z]\s*=\s*)?([-+]?\d+(?:\.\d+)?)(?:\s*[A-Za-z%].*)?$")


def _key(raw: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")


def parse_dimensions(path: str | Path) -> dict[str, float | tuple[float, float] | str]:
    """Return {normalized_key: float | (lo, hi) | str}.

    'Inner diameter: 70.0' -> {'inner_diameter': 70.0}
    '- Top/bottom arm thickness: 10.0' -> {'top_bottom_arm_thickness': 10.0}
    'Nominal concept target load: 5 kg static' -> {...: 5.0}
    'Platform top: Z=100.0' -> {'platform_top': 100.0}
    'Compatible desk range from concept: 20–35' -> {...: (20.0, 35.0)}
    """
    out: dict[str, float | tuple[float, float] | str] = {}
    for line in Path(path).read_text().splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        val = val.strip()
        m = _RANGE.match(val)
        if m:
            out[_key(key)] = (float(m.group(1)), float(m.group(2)))
            continue
        try:
            out[_key(key)] = float(val)
            continue
        except ValueError:
            pass
        m = _LEADING_NUMBER.match(val)
        out[_key(key)] = float(m.group(1)) if m else val
    return out

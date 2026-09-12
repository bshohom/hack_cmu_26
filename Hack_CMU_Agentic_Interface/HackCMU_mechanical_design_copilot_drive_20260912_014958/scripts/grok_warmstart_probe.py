"""One-shot check: can Grok write a valid warm-start script for the cup-holder requirements?

Run from the interface directory with GROK_API_KEY (or XAI_API_KEY) in .env or the shell:

    python scripts/grok_warmstart_probe.py [--out DIR] [--provider grok|mock]

Writes the candidate triple + prompt/response/script files to --out and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from providers import get_provider  # noqa: E402
from schemas import RequirementsUpdate  # noqa: E402
from agents.interaction import InteractionAgent  # noqa: E402
from tools.warmstart import generate_warm_start  # noqa: E402

REQUEST = "I want a cup holder attached to this desk that supports a full 1 L bottle."
ANSWERS = RequirementsUpdate(
    filled_bottle_mass_kg=1.1,
    bottle_diameter_mm=85.0,
    bottle_height_mm=250.0,
    desk_thickness_mm=25.0,
    attachment_method="clamp",
    allowed_contact_region="desk_front_edge",
    attachment_notes="clamp only, no drilling",
    max_protrusion_mm=130.0,
    manufacturing_method="3d_print",
    material="PLA",
    max_part_mass_kg=0.3,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "generated" / "probe_cupholder")
    ap.add_argument("--provider", default="grok")
    ap.add_argument("--name", default="grok_cupholder")
    args = ap.parse_args()

    provider = get_provider(args.provider)
    if not getattr(provider, "configured", False):
        print(json.dumps({"ok": False, "error": provider.not_connected_reason}))
        return 2
    req = InteractionAgent().assess(REQUEST).requirements
    req.description = "desk-mounted cup holder"
    req.user_message = REQUEST
    req.payload.filled_mass_kg = ANSWERS.filled_bottle_mass_kg
    req.object_geometry.bottle_diameter_mm = ANSWERS.bottle_diameter_mm
    req.object_geometry.bottle_height_mm = ANSWERS.bottle_height_mm
    req.environment.desk_thickness_mm = ANSWERS.desk_thickness_mm
    req.attachment.method = ANSWERS.attachment_method
    req.attachment.allowed_contact_region = ANSWERS.allowed_contact_region
    req.attachment.notes = ANSWERS.attachment_notes or ""
    req.design_envelope.max_protrusion_mm = ANSWERS.max_protrusion_mm
    req.manufacturing.method = ANSWERS.manufacturing_method
    req.manufacturing.material = ANSWERS.material
    req.part_mass.max_part_mass_kg = ANSWERS.max_part_mass_kg

    ws = generate_warm_start(req, provider, name=args.name, out_dir=args.out, log=print)
    summary = {
        "ok": ws.ok,
        "model": ws.model,
        "attempts": ws.attempts,
        "latency_s": round(ws.latency_s, 1),
        "out_dir": ws.out_dir,
        "script": ws.script_path,
        "problems": ws.problems,
        "error": ws.error[-800:],
        "candidate": ws.candidate.model_dump() if ws.candidate else None,
        "regions": ws.regions,
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0 if ws.ok else 1


if __name__ == "__main__":
    sys.exit(main())

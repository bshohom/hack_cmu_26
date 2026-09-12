"""Minimal Gradio test UI for surfcap.

Usage:
    python -m surfcap.app [--port 7860]
"""
from __future__ import annotations

import argparse
import shutil
import time
import traceback
from pathlib import Path

import gradio as gr

from .pipeline import render_views, run


def run_pipeline(files, folder_path, table_prompt, card_prompts_str, process_res):
    """Handler for the Run button. Returns (glb_path, target_dict, gallery, summary)."""
    t0 = time.time()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("out") / f"ui_{ts}"
    try:
        folder_path = (folder_path or "").strip()
        if folder_path and Path(folder_path).exists():
            folder = folder_path
        elif files:
            in_dir = out_dir / "input"
            in_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                src = Path(f if isinstance(f, str) else f.name)
                shutil.copy(src, in_dir / src.name)
            folder = str(in_dir)
        else:
            raise ValueError("no input: upload photos or provide a folder path")

        card_prompts = tuple(
            p.strip() for p in (card_prompts_str or "").split(",") if p.strip()
        ) or ("card",)

        target = run(
            folder,
            out_dir,
            card_prompts=card_prompts,
            table_prompt=table_prompt or "table",
            process_res=int(process_res),
            debug=True,
        )
        try:
            render_views(out_dir)
        except Exception:
            pass  # world_top/world_side plots are best-effort

        dbg = out_dir / "debug"
        gallery = [str(dbg / n) for n in ("world_top.png", "world_side.png") if (dbg / n).exists()]
        mask_dir = dbg / "masks"
        if mask_dir.exists():
            gallery += [str(p) for p in sorted(mask_dir.glob("*_overlay.jpg"))[:4]]

        sc = target.scale or {}
        lines = [
            f"scale factor     : {sc.get('factor')}",
            f"scale rms_mm     : {sc.get('rms_mm')}",
            f"n_views_with_ref : {sc.get('n_views_with_ref')}  reliable={sc.get('reliable')}",
            f"card prompt used : {sc.get('card_prompt_used')}",
            "-" * 40,
        ]
        for s in target.surfaces:
            lines.append(
                f"{s.id:<10} {s.role:<8} normal={s.normal} "
                f"extent_m={s.extent_m} rms_mm={s.planarity_rms_mm:.2f}"
            )
        if not target.surfaces:
            lines.append("(no surfaces)")
        lines.append("-" * 40)
        lines.append(f"warnings ({len(target.warnings)}):")
        for w in target.warnings:
            lines.append(f"  - {w}")
        lines.append(f"total time       : {time.time() - t0:.1f}s")

        glb_path = out_dir / "target.glb"
        return (
            str(glb_path) if glb_path.exists() else None,
            target.to_dict(),
            gallery,
            "\n".join(lines),
        )
    except Exception:
        return None, {}, [], traceback.format_exc()


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="surfcap test UI") as demo:
        gr.Markdown("# surfcap — minimal test UI")
        with gr.Row():
            files = gr.File(file_count="multiple", file_types=["image"], label="Photos")
            folder = gr.Textbox(label="or folder path on this machine", value="data/table_a")
        with gr.Row():
            table_prompt = gr.Textbox(label="table prompt", value="table")
            card_prompts = gr.Textbox(
                label="card prompts (comma-separated)",
                value="card, credit card, membership card",
            )
            process_res = gr.Slider(512, 1024, value=768, step=128, label="process_res")
        run_btn = gr.Button("Run")
        with gr.Row():
            model3d = gr.Model3D(label="target.glb")
            out_json = gr.JSON(label="target.json")
        gallery = gr.Gallery(label="debug views")
        summary = gr.Textbox(label="summary", lines=20, max_lines=40)

        run_btn.click(
            run_pipeline,
            inputs=[files, folder, table_prompt, card_prompts, process_res],
            outputs=[model3d, out_json, gallery, summary],
        )
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=False)


if __name__ == "__main__":
    main()

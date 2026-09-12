"""
Text-prompted instance segmentation with SAM 3 (`facebook/sam3`) for surfcap.

LOAD PATH THAT WORKED (transformers 5.5.4, verified 2026-09-11 on this box):

    model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.bfloat16,
                                      output_loading_info=True)   # -> (model, info)
    # info == {'missing_keys': [], 'unexpected_keys': [], 'mismatched_keys': [], 'error_msgs': []}
    processor = Sam3Processor.from_pretrained("facebook/sam3")

i.e. the PRIMARY path is clean and the plan's `Sam3VideoModel` workaround is NOT needed on
transformers 5.5.4.  The cached `config.json` really is `model_type: sam3_video` /
`architectures: ["Sam3VideoModel"]` with weights prefixed `detector_model.`, but 5.5.4's
`Sam3Model` already handles both:

    class Sam3Model(Sam3PreTrainedModel):
        base_model_prefix = "detector_model"
        _keys_to_ignore_on_load_unexpected = [r"^tracker_model.", r"^tracker_neck."]
        def __init__(self, config):
            # loading from a sam3_video config
            if hasattr(config, "detector_config") and config.detector_config is not None: ...

so the `detector_model.` prefix is stripped and the tracker weights are dropped for us.
The `Sam3VideoModel` -> detector-submodule fallback is still implemented below (and the
hand-built `Sam3Processor(Sam3ImageProcessor(...), AutoTokenizer(...))` fallback) but on this
box neither fires.  `Sam3Processor.from_pretrained` yields size 1008x1008, mask_size 288x288,
image_mean/std (0.5, 0.5, 0.5) -- exactly the hand-built fallback's numbers.

POST-PROCESSING API USED (transformers/models/sam3/processing_sam3.py:595):

    res = processor.post_process_instance_segmentation(
        outputs, threshold=threshold, mask_threshold=0.5,
        target_sizes=inputs["original_sizes"].tolist())[0]
    # -> {"scores": [K], "boxes": [K,4] xyxy, "masks": [K,H,W] bool}

TEXT-FEATURE REUSE (nice-to-have, implemented): `Sam3Model.get_text_features(input_ids,
attention_mask).pooler_output` is cached per prompt and passed back as `text_embeds=` (with the
matching `attention_mask=`), so the CLIP text tower runs once per prompt instead of once per
image.  `Sam3Model.forward` accepts exactly one of `input_ids` / `text_embeds`.

Units/conventions: see surfcap/types.py.  Nothing here raises; every problem becomes a string
appended to the returned `warnings` list.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from surfcap.types import Masks

_CARD_ASPECT = 85.60 / 53.98  # 1.5857... ISO ID-1

# instance-selection constants (from the contract)
TABLE_AREA_FRAC_OF_LARGEST = 0.40
TABLE_UNION_MAX_COVERAGE = 0.85
TABLE_SUSPICIOUS_LO = 0.05
TABLE_SUSPICIOUS_HI = 0.80
CARD_AREA_MIN_FRAC = 0.0005  # 0.05 %
CARD_AREA_MAX_FRAC = 0.20  # 20 %

# Records which load path actually ran, for the caller / debug print.
LOAD_PATH: str = "not loaded"


# --------------------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------------------
def load_sam3(device: str = "cuda", dtype=torch.bfloat16):
    """Load SAM 3 detector + processor.  Returns (model, processor).

    Tries `Sam3Model.from_pretrained` first and verifies via `output_loading_info=True` that
    nothing is missing/mismatched; falls back to `Sam3VideoModel` -> detector submodule.
    """
    global LOAD_PATH
    from transformers import Sam3Model, Sam3Processor

    repo = "facebook/sam3"
    model = None
    try:
        model, info = Sam3Model.from_pretrained(repo, dtype=dtype, output_loading_info=True)
        missing = list(info.get("missing_keys", []))
        mismatched = list(info.get("mismatched_keys", []))
        errs = list(info.get("error_msgs", []))
        # unexpected_keys are tolerable (tracker weights) but missing ones are not
        if missing or mismatched or errs:
            raise RuntimeError(
                f"Sam3Model load incomplete: {len(missing)} missing, "
                f"{len(mismatched)} mismatched, errors={errs[:2]}"
            )
        LOAD_PATH = 'Sam3Model.from_pretrained("facebook/sam3", dtype=..) [clean, 0 missing keys]'
    except Exception as e:  # noqa: BLE001 - never crash on a load path
        print(f"[segment] Sam3Model path failed ({e}); falling back to Sam3VideoModel")
        model = None

    if model is None:
        from transformers import Sam3VideoModel

        video = Sam3VideoModel.from_pretrained(repo, dtype=dtype)
        det_name = None
        for name, child in video.named_children():
            if "detector" in name.lower():
                det_name = name
                break
        if det_name is None:  # last resort: first child that is not a tracker
            for name, _ in video.named_children():
                if "track" not in name.lower():
                    det_name = name
                    break
        if det_name is None:
            raise RuntimeError(
                "no detector submodule in Sam3VideoModel; children="
                f"{[n for n, _ in video.named_children()]}"
            )
        model = getattr(video, det_name)
        # drop the tracker halves so they never reach the GPU
        for name, _ in list(video.named_children()):
            if name != det_name:
                setattr(video, name, None)
        del video
        gc.collect()
        LOAD_PATH = f'Sam3VideoModel.from_pretrained("facebook/sam3").{det_name} [fallback]'

    try:
        processor = Sam3Processor.from_pretrained(repo)
    except Exception as e:  # noqa: BLE001
        print(f"[segment] Sam3Processor.from_pretrained failed ({e}); hand-building processor")
        from transformers import AutoTokenizer, Sam3ImageProcessorFast

        processor = Sam3Processor(
            Sam3ImageProcessorFast(
                size={"height": 1008, "width": 1008},
                image_mean=[0.5] * 3,
                image_std=[0.5] * 3,
                mask_size={"height": 288, "width": 288},
            ),
            AutoTokenizer.from_pretrained(repo),
        )
        LOAD_PATH += " + hand-built Sam3Processor"

    model = model.to(device).eval()
    return model, processor


def free_gpu(*objs) -> None:
    """Drop references, collect, empty the CUDA cache.  Caller must also drop its own refs."""
    for o in objs:
        try:
            if isinstance(o, torch.nn.Module):
                o.to("cpu")
        except Exception:  # noqa: BLE001
            pass
    try:
        del objs
    except Exception:  # noqa: BLE001
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# --------------------------------------------------------------------------------------
# instance selection
# --------------------------------------------------------------------------------------
def _mask_aspect(mask: np.ndarray) -> float:
    """long/short of cv2.minAreaRect over the mask pixels; +inf if degenerate."""
    ys, xs = np.nonzero(mask)
    if xs.size < 3:
        return float("inf")
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    (_, (w, h), _) = cv2.minAreaRect(pts)
    lo, hi = min(w, h), max(w, h)
    if lo <= 1e-6:
        return float("inf")
    return float(hi / lo)


def _select_table(masks: np.ndarray, scores: np.ndarray, hw: tuple[int, int]):
    """Union of big instances; collapse to the largest if the union swallows the frame."""
    h, w = hw
    frame_area = float(h * w)
    if masks.shape[0] == 0:
        return np.zeros((h, w), dtype=bool), 0.0
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1).astype(np.float64)
    largest = int(np.argmax(areas))
    if areas[largest] <= 0:
        return np.zeros((h, w), dtype=bool), 0.0
    keep = np.nonzero(areas >= TABLE_AREA_FRAC_OF_LARGEST * areas[largest])[0]
    union = np.any(masks[keep], axis=0)
    if union.sum() / frame_area > TABLE_UNION_MAX_COVERAGE:
        keep = np.array([largest])
        union = masks[largest].copy()
    score = float(np.max(scores[keep])) if keep.size else 0.0
    return union.astype(bool), score


def _select_card(masks: np.ndarray, scores: np.ndarray, hw: tuple[int, int]):
    """Best score * exp(-|aspect - 1.586|) among area-plausible instances."""
    h, w = hw
    frame_area = float(h * w)
    if masks.shape[0] == 0:
        return np.zeros((h, w), dtype=bool), 0.0, 0
    best_i, best_q = -1, -1.0
    n_pass = 0
    for i in range(masks.shape[0]):
        frac = float(masks[i].sum()) / frame_area
        if not (CARD_AREA_MIN_FRAC <= frac <= CARD_AREA_MAX_FRAC):
            continue
        aspect = _mask_aspect(masks[i])
        if not np.isfinite(aspect):
            continue
        n_pass += 1
        q = float(scores[i]) * float(np.exp(-abs(aspect - _CARD_ASPECT)))
        if q > best_q:
            best_q, best_i = q, i
    if best_i < 0:
        return np.zeros((h, w), dtype=bool), 0.0, 0
    return masks[best_i].astype(bool), float(scores[best_i]), n_pass


# --------------------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------------------
def segment_images(
    frames,
    prompts: dict[str, str] = {"table": "table", "card": "credit card"},
    threshold: float = 0.5,
    model=None,
    processor=None,
) -> tuple[Masks, list[str]]:
    """Run SAM 3 per frame per prompt and reduce to one table mask + one card mask per view.

    Returns (Masks, warnings).  Masks.table / Masks.card are [N,H,W] bool at frame.rgb
    resolution.  Never raises; failures become warnings and all-False masks.
    """
    from PIL import Image

    warns: list[str] = []
    n = len(frames)
    if n == 0:
        empty = np.zeros((0, 1, 1), dtype=bool)
        return Masks(empty, empty, [], []), ["zero frames"]

    h, w = frames[0].rgb.shape[:2]
    if any(f.rgb.shape[:2] != (h, w) for f in frames):
        warns.append("frames_have_mixed_resolutions")

    owns_model = model is None or processor is None
    if owns_model:
        model, processor = load_sam3()
    device = next(model.parameters()).device
    mdtype = next(model.parameters()).dtype

    table_masks: list[np.ndarray] = []
    card_masks: list[np.ndarray] = []
    table_scores: list[float] = []
    card_scores: list[float] = []

    # cache text features per prompt string (reuse across images)
    text_cache: dict[str, tuple] = {}

    def _text_embeds(prompt: str):
        if prompt in text_cache:
            return text_cache[prompt]
        try:
            tenc = processor(text=prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.get_text_features(
                    input_ids=tenc["input_ids"], attention_mask=tenc.get("attention_mask")
                )
            emb = out.pooler_output if hasattr(out, "pooler_output") else out[1]
            text_cache[prompt] = (emb, tenc.get("attention_mask"))
        except Exception:  # noqa: BLE001 - fall back to per-image tokenisation
            text_cache[prompt] = None
        return text_cache[prompt]

    table_prompt = prompts.get("table", "table")
    card_prompt = prompts.get("card", "credit card")

    for i, fr in enumerate(frames):
        fh, fw = fr.rgb.shape[:2]
        pil = Image.fromarray(fr.rgb)
        per_prompt: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for key, prompt in (("table", table_prompt), ("card", card_prompt)):
            try:
                inputs = processor(images=pil, text=prompt, return_tensors="pt").to(device)
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(mdtype)
                cached = _text_embeds(prompt)
                call = dict(inputs)
                if cached is not None:
                    emb, amask = cached
                    call.pop("input_ids", None)
                    call["text_embeds"] = emb.to(mdtype)
                    if amask is not None:
                        call["attention_mask"] = amask
                call.pop("original_sizes", None)
                with torch.no_grad():
                    if device.type == "cuda":
                        with torch.autocast("cuda", dtype=mdtype):
                            outputs = model(**call)
                    else:
                        outputs = model(**call)
                res = processor.post_process_instance_segmentation(
                    outputs,
                    threshold=threshold,
                    mask_threshold=0.5,
                    target_sizes=inputs["original_sizes"].tolist(),
                )[0]
                m = res["masks"]
                s = res["scores"]
                m_np = (
                    m.detach().to(torch.uint8).cpu().numpy().astype(bool)
                    if torch.is_tensor(m)
                    else np.asarray(m, dtype=bool)
                )
                s_np = (
                    s.detach().float().cpu().numpy()
                    if torch.is_tensor(s)
                    else np.asarray(s, dtype=np.float32)
                )
                if m_np.ndim == 2:
                    m_np = m_np[None]
                if m_np.shape[0] and m_np.shape[-2:] != (fh, fw):
                    m_np = np.stack(
                        [
                            cv2.resize(
                                mm.astype(np.uint8), (fw, fh), interpolation=cv2.INTER_NEAREST
                            ).astype(bool)
                            for mm in m_np
                        ]
                    )
                per_prompt[key] = (m_np, s_np)
            except Exception as e:  # noqa: BLE001
                warns.append(f"segment_failed_view={i}:{key}:{type(e).__name__}")
                per_prompt[key] = (np.zeros((0, fh, fw), dtype=bool), np.zeros((0,), np.float32))

        tm, ts = _select_table(*per_prompt["table"], (fh, fw))
        cm, cs, n_card_pass = _select_card(*per_prompt["card"], (fh, fw))

        cov = float(tm.sum()) / float(fh * fw)
        if not tm.any():
            warns.append(f"table_mask_empty_view={i}")
        elif cov < TABLE_SUSPICIOUS_LO or cov > TABLE_SUSPICIOUS_HI:
            warns.append(f"table_mask_suspicious_view={i}")
        if not cm.any():
            warns.append(f"card_not_found_view={i}")
        if n_card_pass >= 2:
            warns.append(f"multiple_card_candidates_view={i}")

        table_masks.append(tm)
        card_masks.append(cm)
        table_scores.append(ts)
        card_scores.append(cs)

    # frames may differ in resolution (mixed aspect ratios); pad to the max box so the
    # contract's [N,H,W] array still holds.  Same-size frames (the normal case) are untouched.
    max_h = max(m.shape[0] for m in table_masks)
    max_w = max(m.shape[1] for m in table_masks)

    def _pad(m):
        if m.shape == (max_h, max_w):
            return m
        out = np.zeros((max_h, max_w), dtype=bool)
        out[: m.shape[0], : m.shape[1]] = m
        return out

    table_arr = np.stack([_pad(m) for m in table_masks]).astype(bool)
    card_arr = np.stack([_pad(m) for m in card_masks]).astype(bool)

    k = int(sum(1 for m in card_masks if m.any()))
    warns.append(f"card_found_in_k_views={k}")
    if not table_arr.any():
        warns.append("table_mask_empty")

    if owns_model:
        free_gpu(model, processor)
        del model, processor

    return Masks(table_arr, card_arr, table_scores, card_scores), warns


# --------------------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------------------
def detect_card_quad(
    gray,
    aspect_lo: float = 1.35,
    aspect_hi: float = 2.60,
    area_lo_frac: float = 0.0003,
    area_hi_frac: float = 0.30,
    min_contrast: float = 20.0,
):
    """(F1-D3) Prompt-free card detector: the best high-contrast ID-1 quadrilateral.

    SAM 3 found the card in 0/11 views on the grey window-ledge scene, so the card
    has to be findable without a text prompt.  Canny -> contours -> 4-vertex convex
    ``approxPolyDP`` -> ``minAreaRect`` with ID-1 aspect (1.586 +/- ~0.14) and a
    grey-level step between the quad's interior and a 10 px outer ring (works
    for white-on-dark and dark-on-white alike).

    (F2-1) Gates loosened after the ledge diagnosis: the card IS recovered as a
    convex 4-gon there (fill 0.89-1.02) but was rejected twice over -- its
    *perspective* aspect reaches 2.23-2.40 (foreshortened short edge, not 1.586)
    and a white card on a light-grey sill only steps 8-28 grey levels, not 50.
    So aspect -> [1.35, 2.60], contrast -> >= 20, area -> <= 30 %.

    Returns ``(mask[H,W] bool, info dict)`` or ``(None, None)``.
    """
    import cv2

    g = np.asarray(gray)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8)
    h, w = g.shape[:2]
    frame_area = float(h * w)
    gb = cv2.GaussianBlur(g, (5, 5), 0)
    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

    best = None
    for lo, hi in ((50, 150), (30, 90), (80, 200)):
        edges = cv2.dilate(cv2.Canny(gb, lo, hi), np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cnt.shape[0] < 4:
                continue
            a = float(cv2.contourArea(cnt))
            if not (area_lo_frac * frame_area <= a <= area_hi_frac * frame_area):
                continue
            peri = float(cv2.arcLength(cnt, True))
            if peri <= 0:
                continue
            ap = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if ap.shape[0] != 4 or not cv2.isContourConvex(ap):
                continue
            (rw, rh) = cv2.minAreaRect(ap)[1]
            if min(rw, rh) < 12.0 or rw * rh <= 0:
                continue
            asp = float(max(rw, rh) / max(min(rw, rh), 1e-6))
            if not (aspect_lo <= asp <= aspect_hi):
                continue
            if a / float(rw * rh) < 0.75:
                continue
            inner = np.zeros((h, w), np.uint8)
            cv2.drawContours(inner, [ap], -1, 255, -1)
            ring = (cv2.dilate(inner, ring_k) > 0) & (inner == 0)
            if int(ring.sum()) < 50:
                continue
            mi = float(g[inner > 0].mean())
            mo = float(g[ring].mean())
            contrast = abs(mi - mo)
            if contrast < min_contrast:
                continue
            score = (contrast * (a / frame_area) ** 0.25
                     * float(np.exp(-0.5 * abs(asp - _CARD_ASPECT))))
            if best is None or score > best[0]:
                best = (score, inner > 0,
                        {"contrast": round(contrast, 1), "aspect": round(asp, 3),
                         "area_frac": round(a / frame_area, 5), "canny": [lo, hi],
                         "score": round(score, 2)})
        if best is not None:
            break
    if best is None:
        return None, None
    return best[1], best[2]


def _overlay(rgb: np.ndarray, table: np.ndarray, card: np.ndarray) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    if table.any():
        out[table] = 0.55 * out[table] + 0.45 * np.array([0, 255, 0], np.float32)
    if card.any():
        out[card] = 0.45 * out[card] + 0.55 * np.array([255, 0, 0], np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(prog="surfcap.segment")
    ap.add_argument("folder")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--table-prompt", default="table")
    ap.add_argument("--card-prompt", default="credit card")
    ap.add_argument("--out", default="out/debug/masks")
    args = ap.parse_args()

    from surfcap.io_images import load_folder

    frames, io_warns = load_folder(args.folder, max_images=args.n, min_keep=min(args.n, 8))
    frames = frames[: args.n]
    print(f"loaded {len(frames)} frame(s); io warnings: {io_warns}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model, processor = load_sam3()
    print(f"LOAD_PATH: {LOAD_PATH}  ({time.time() - t0:.1f}s)")

    t1 = time.time()
    masks, warns = segment_images(
        frames,
        prompts={"table": args.table_prompt, "card": args.card_prompt},
        threshold=args.threshold,
        model=model,
        processor=processor,
    )
    dt = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'view':>4} {'stem':<28} {'table%':>7} {'t_score':>8} {'card%':>7} {'c_score':>8}")
    for i, fr in enumerate(frames):
        h, w = fr.rgb.shape[:2]
        tcov = 100.0 * masks.table[i].sum() / (h * w)
        ccov = 100.0 * masks.card[i].sum() / (h * w)
        stem = Path(fr.path).stem
        print(
            f"{i:>4} {stem[:28]:<28} {tcov:>7.2f} {masks.table_scores[i]:>8.3f} "
            f"{ccov:>7.3f} {masks.card_scores[i]:>8.3f}"
        )
        cv2.imwrite(str(outdir / f"{stem}_table.png"), masks.table[i].astype(np.uint8) * 255)
        cv2.imwrite(str(outdir / f"{stem}_card.png"), masks.card[i].astype(np.uint8) * 255)
        ov = _overlay(fr.rgb, masks.table[i], masks.card[i])
        cv2.imwrite(str(outdir / f"{stem}_overlay.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

    print(f"\nwall: {dt:.2f}s total, {dt / max(1, len(frames)):.2f}s/image (2 prompts each)")
    print(f"peak VRAM allocated: {peak:.2f} GB")
    print(f"warnings: {warns}")
    print(f"masks written to {outdir.resolve()}")

    free_gpu(model, processor)
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"mem_get_info after free_gpu: free={free_b / 1e9:.2f} GB / total={total_b / 1e9:.2f} GB")


if __name__ == "__main__":
    main()

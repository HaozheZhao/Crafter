#!/usr/bin/env python3
"""Run Crafter over a benchmark and write one image per sample.

    export OPENROUTER_API_KEY="sk-or-..."

    # CraftBench (bundled in this repo)
    python inference.py --bench craftbench --out runs/crafter_cb

    # PaperBanana-Bench (point at the official test.json)
    python inference.py --bench paperbanana --pb-data path/to/test.json \
        --out runs/crafter_pb

Score the outputs:

    # CraftBench
    python -m craftbench.evaluation.run_eval --runs runs/crafter_cb --out cb.json
    # PaperBanana — use the official PaperBanana evaluation on runs/crafter_pb
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from crafter.generation.core.config import CraftConfig
from crafter.generation.craft.session import CraftSession, CraftInput

REPO = Path(__file__).resolve().parent

# style -> communicative role
ROLE = {"academic": "academic", "academic_clean": "academic",
        "poster": "poster", "infographic": "infographic"}
# edit task -> reference-image role hint
REFER_ROLE = {"inpaint": "preserve_partial", "keyelems": "use_elements",
              "sketch": "refine_sketch"}


def _figure_type(task: str, style: str) -> str:
    if task == "t2i" and style in ("academic", "academic_clean"):
        return "method_pipeline"
    return task or "method_pipeline"


def _craft_one(sample: dict, out_dir: Path, cfg_path: str,
               max_iters: int, num_variants: int) -> dict:
    sid = sample["id"]
    png = out_dir / f"{sid}.png"
    if png.exists():
        return {"id": sid, "status": "skip"}
    t0 = time.time()
    try:
        cfg = CraftConfig.from_yaml(cfg_path or None,
                                    output_dir=str(out_dir / sid),
                                    max_iterations=max_iters,
                                    num_variants=num_variants)
        cfg.ensure_dirs()
        task = sample.get("task", "t2i")
        style = sample.get("style", "academic")
        refer = sample.get("input_image") or ""
        refer_paths = [refer] if refer else []
        ci = CraftInput(
            paper_text=(sample.get("paper_context", "") or "")[:30000],
            description=(sample.get("caption", "") or sample.get("instruction", ""))[:600],
            figure_type=_figure_type(task, style),
            venue="neurips",
            role=ROLE.get(style, "academic"),
            reference_paths=refer_paths,
            refer_image_role=REFER_ROLE.get(task, "") if refer else "",
            max_iterations=max_iters, num_variants=num_variants,
            output_path=str(png),
        )
        result = CraftSession(cfg).craft(ci)
        src = result.final_image_path or result.best_image_path
        if src and Path(src).exists() and Path(src) != png:
            png.write_bytes(Path(src).read_bytes())
        ok = png.exists()
        return {"id": sid, "status": "ok" if ok else "no_image",
                "secs": round(time.time() - t0, 1)}
    except Exception as e:
        return {"id": sid, "status": "error", "error": f"{type(e).__name__}: {e}",
                "secs": round(time.time() - t0, 1)}


def load_craftbench(limit: int):
    """Load CraftBench from the HuggingFace Hub (BleachNick/CraftBench)."""
    from datasets import load_dataset
    import tempfile
    ds = load_dataset("BleachNick/CraftBench", split="test")
    cache_dir = Path(tempfile.gettempdir()) / "craftbench_inputs"
    cache_dir.mkdir(exist_ok=True)
    out = []
    for s in ds:
        inp_path = None
        if s["input_image"]:
            inp_path = cache_dir / f"{s['id']}_input.png"
            s["input_image"].save(inp_path)
        out.append({
            "id": s["id"], "task": s["task"], "style": s["style"],
            "caption": s["caption"], "paper_context": s["paper_context"],
            "instruction": s["instruction"],
            "input_image": str(inp_path) if inp_path else None,
        })
    return out[:limit] if limit else out


def load_paperbanana(pb_data: str, limit: int):
    raw = json.loads(Path(pb_data).read_text())
    root = Path(pb_data).resolve().parent
    out = []
    for s in raw:
        out.append({
            "id": f"pb_{s['id']}", "task": "t2i", "style": "academic",
            "caption": s.get("visual_intent", ""),
            "paper_context": s.get("content", ""),
            "instruction": "",
            "input_image": None,
        })
    return out[:limit] if limit else out


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Crafter over a benchmark.")
    ap.add_argument("--bench", choices=["craftbench", "paperbanana"], default="craftbench")
    ap.add_argument("--out", required=True, help="output dir; writes <id>.png per sample")
    ap.add_argument("--pb-data", default="", help="PaperBanana official test.json (for --bench paperbanana)")
    ap.add_argument("--config", default="", help="config yaml (default configs/default.yaml)")
    ap.add_argument("--limit", type=int, default=0, help="cap number of samples (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--num-variants", type=int, default=3)
    a = ap.parse_args()

    if a.bench == "craftbench":
        samples = load_craftbench(a.limit)
    else:
        if not a.pb_data:
            sys.exit("--pb-data is required for --bench paperbanana")
        samples = load_paperbanana(a.pb_data, a.limit)

    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[inference] bench={a.bench} n={len(samples)} workers={a.workers} out={out_dir}")

    n_ok = n_skip = n_err = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_craft_one, s, out_dir, a.config, a.max_iters, a.num_variants): s["id"]
                for s in samples}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            n_ok += r["status"] == "ok"
            n_skip += r["status"] == "skip"
            n_err += r["status"] in ("error", "no_image")
            if r["status"] in ("error", "no_image"):
                print(f"  {r['id']}: {r['status']} {r.get('error','')}")
            if i % 10 == 0 or i == len(samples):
                print(f"  [{i}/{len(samples)}] ok={n_ok} skip={n_skip} err={n_err}", flush=True)
    print(f"[inference] done: ok={n_ok} skip={n_skip} err={n_err} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

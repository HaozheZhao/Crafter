#!/usr/bin/env python3
"""Evaluate a system's CraftBench outputs against the human-drawn targets.

The 279-sample CraftBench dataset is loaded directly from the HuggingFace Hub
(`BleachNick/CraftBench`). Place your generated figures in a directory, one per
sample, named by sample id:

    runs/my_system/craftbench-0001.png
    runs/my_system/craftbench-0002.png
    ...

Then:

    export OPENROUTER_API_KEY="sk-or-..."
    python -m craftbench.evaluation.run_eval --runs runs/my_system --out my_system.json

A missing or unreadable generation counts as a Human win. The script reports the
lenient win-rate overall and per task.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import judge

VERDICT_SCORE = {"Model": 100.0, "Tie": 50.0, "Human": 0.0}
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")
HF_DATASET = "BleachNick/CraftBench"


def find_candidate(runs: Path, sid: str):
    for ext in IMG_EXTS:
        p = runs / f"{sid}{ext}"
        if p.exists():
            return p
    return None


def _materialise(img, dst: Path) -> Path:
    """Save a HuggingFace ``Image`` feature (PIL) to ``dst`` and return the path."""
    if img is None:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)
    return dst


def score_sample(entry: dict, runs: Path, cache: Path):
    sid = entry["id"]
    rec = {
        "task": entry["task"], "style": entry["style"],
        "caption": entry.get("caption", ""), "paper_context": entry.get("paper_context", ""),
        "instruction": entry.get("instruction", ""), "masked_region": entry.get("masked_region", ""),
        "input_path": str(_materialise(entry.get("input_image"), cache / "inputs" / f"{sid}.png"))
                       if entry.get("input_image") else None,
    }
    target_path = _materialise(entry["target_image"], cache / "gt" / f"{sid}.png")
    cand = find_candidate(runs, sid)
    if cand is None:
        return {"id": sid, "task": rec["task"], "verdict": "Human", "missing": True}
    try:
        sm = judge.score_image(rec, str(cand))
        sg = judge.score_image(rec, str(target_path))
    except Exception as e:
        return {"id": sid, "task": rec["task"], "verdict": "Human",
                "missing": True, "reason": f"{type(e).__name__}: {e}"}
    tm, tg = judge.weighted_total(sm["scores"]), judge.weighted_total(sg["scores"])
    if tm is None or tg is None:
        return {"id": sid, "task": rec["task"], "verdict": "Human", "missing": True,
                "reason": "judge returned no usable scores"}
    return {"id": sid, "task": rec["task"],
            "verdict": judge.verdict(tm - tg), "margin": round(tm - tg, 3),
            "candidate_scores": sm["scores"], "target_scores": sg["scores"], "missing": False}


def aggregate(results):
    def winrate(rs):
        return round(sum(VERDICT_SCORE[r["verdict"]] for r in rs) / len(rs), 2) if rs else 0.0
    by_task = {}
    for r in results:
        by_task.setdefault(r["task"], []).append(r)
    return {
        "overall": winrate(results),
        "by_task": {t: winrate(rs) for t, rs in sorted(by_task.items())},
        "n": len(results),
        "missing": sum(1 for r in results if r.get("missing")),
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate CraftBench outputs.")
    ap.add_argument("--runs", required=True, help="dir with generated images named <id>.png")
    ap.add_argument("--out", default="craftbench_result.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="optionally cap to the first N samples")
    a = ap.parse_args()

    from datasets import load_dataset
    print(f"loading {HF_DATASET} ...", flush=True)
    ds = load_dataset(HF_DATASET, split="test")
    samples = ds.select(range(a.limit)) if a.limit else ds
    cache = Path(tempfile.mkdtemp(prefix="craftbench_eval_"))

    runs = Path(a.runs)
    results = {}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(score_sample, dict(samples[i]), runs, cache): samples[i]["id"]
                for i in range(len(samples))}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            results[r["id"]] = r
            if i % 25 == 0 or i == len(samples):
                print(f"  [{i}/{len(samples)}]", flush=True)

    ordered = [results[s["id"]] for s in samples]
    summary = aggregate(ordered)
    out = {"summary": summary, "results": ordered}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print("\n=== CraftBench ===")
    print(f"  overall : {summary['overall']:.1f}   (n={summary['n']}, missing={summary['missing']})")
    for t, v in summary["by_task"].items():
        print(f"  {t:<9}: {v:.1f}")
    print(f"\nwritten -> {a.out}")


if __name__ == "__main__":
    main()

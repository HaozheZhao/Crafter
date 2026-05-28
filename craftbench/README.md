# CraftBench

279 samples spanning three figure types (academic / poster / infographic) and
four input conditions (text-to-image, mask completion, key-element composition,
sketch refinement), each with a human-drawn target.

The full dataset is hosted on HuggingFace:
**[BleachNick/CraftBench](https://huggingface.co/datasets/BleachNick/CraftBench)**.

```python
from datasets import load_dataset
ds = load_dataset("BleachNick/CraftBench", split="test")
```

## What's in this folder

This folder is a minimal in-repo stub:

- `manifest.json` + `samples/` + `images/` — three illustrative samples (one
  per task) so the on-disk layout and `paper_context` / `instruction` field
  formats are visible without downloading the full dataset.
- `evaluation/` — the self-contained scoring scripts. They load the full
  279-sample dataset from HuggingFace at run time.

## Inference + evaluation

```bash
# Generate Crafter outputs over the bench. Writes <id>.png per sample.
python inference.py --bench craftbench --out runs/crafter_cb

# Score against the human-drawn targets.
python -m craftbench.evaluation.run_eval --runs runs/crafter_cb --out cb.json
```

`run_eval` reports an overall win-rate and a per-task breakdown. A missing or
unreadable generation counts as a Human win.

## License

MIT — see [LICENSE](LICENSE).

#!/usr/bin/env python3
"""End-to-end Crafter demo.

    export OPENROUTER_API_KEY="sk-or-..."

    # 1. Generate a figure from a paper PDF + an instruction.
    python demo.py --paper paper.pdf \
                   --instruction "Draw the overall architecture of our method." \
                   --out figure.png

    # 2. Continue into the editor and produce an editable SVG.
    python demo.py --paper paper.pdf --instruction "..." --out figure.png --edit

A reference image (sketch, partial figure, or icon collage) is optional:

    python demo.py --paper paper.pdf --instruction "..." \
                   --reference sketch.png --out figure.png
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from crafter.editor import Editor
from crafter.generation.core.config import CraftConfig
from crafter.generation.craft.session import CraftInput, CraftSession


def _extract_pdf_text(pdf: Path) -> str:
    """Pull text from a PDF; truncates to the first 32 K characters."""
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise SystemExit(
            "Reading PDFs requires `pypdf`; install with `pip install pypdf`."
        ) from e
    reader = PdfReader(str(pdf))
    out = []
    for page in reader.pages:
        try:
            out.append(page.extract_text() or "")
        except Exception:
            continue
    return ("\n\n".join(out))[:32000]


def _load_paper_text(spec: str) -> str:
    p = Path(spec).expanduser()
    if not p.exists():
        # Treat as inline text.
        return spec
    if p.suffix.lower() == ".pdf":
        return _extract_pdf_text(p)
    return p.read_text(errors="ignore")


def generate(paper_text: str, instruction: str, out: Path,
             reference: str | None, config_path: str | None) -> Path:
    cfg = CraftConfig.from_yaml(config_path or None,
                                output_dir=str(out.parent / f"_{out.stem}_run"))
    cfg.ensure_dirs()
    ci = CraftInput(
        paper_text=paper_text,
        description=instruction,
        figure_type="method_pipeline",
        reference_paths=[reference] if reference else [],
        refer_image_role="refine_sketch" if reference else "",
        output_path=str(out),
    )
    result = CraftSession(cfg).craft(ci)
    src = result.final_image_path or result.best_image_path
    if not src or not Path(src).exists():
        raise RuntimeError("Crafter did not produce an image.")
    out.write_bytes(Path(src).read_bytes())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Crafter end-to-end demo.")
    ap.add_argument("--paper", required=True,
                    help="paper PDF, text file, or inline text")
    ap.add_argument("--instruction", default="",
                    help="what the figure should show / how it should look "
                         "(inline text, or pass --instruction-file instead)")
    ap.add_argument("--instruction-file", default="",
                    help="path to a text file containing the instruction "
                         "(takes precedence over --instruction)")
    ap.add_argument("--out", required=True, help="output image path (PNG)")
    ap.add_argument("--reference", default="",
                    help="optional reference image (sketch / icons / partial)")
    ap.add_argument("--edit", action="store_true",
                    help="after generation, convert the figure to an editable SVG")
    ap.add_argument("--sam-only", action="store_true",
                    help="(with --edit) skip the gpt-image-2 extraction phase")
    ap.add_argument("--config", default="", help="config yaml (default configs/default.yaml)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    instruction = (Path(a.instruction_file).read_text()
                   if a.instruction_file else a.instruction)
    if not instruction.strip():
        raise SystemExit("--instruction or --instruction-file is required")
    paper_text = _load_paper_text(a.paper)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[demo] generating → {out}")
    generate(paper_text, instruction, out, a.reference or None, a.config or None)
    print(f"[demo] wrote {out}")

    if a.edit:
        from crafter.editor.config import Config, DEFAULT_CONFIG
        from dataclasses import replace
        try:
            ed_config = Config.from_yaml(a.config or None)
        except Exception:
            ed_config = DEFAULT_CONFIG
        if a.sam_only:
            ed_config = replace(ed_config,
                                extraction=replace(ed_config.extraction, use_gpt_image2=False))
        edit_out = out.parent / f"{out.stem}_editable"
        print(f"[demo] converting to editable SVG → {edit_out}/  (sam-only={a.sam_only})")
        Editor(config=ed_config).run(str(out), str(edit_out))
        print(f"[demo] editable assets in {edit_out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

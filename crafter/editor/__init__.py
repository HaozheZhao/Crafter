"""Editor — raster-to-vector conversion with the harness abstraction.

Three phases (matching paper §3.3 \\Editor: Harness for Raster-to-Vector
Conversion):

  extraction  — VLM designer D writes a keep/delete plan;
                instructable editor E executes it at pixel level;
                verifier V inspects and triggers revisions for T≤3 rounds.

  processing  — caption, ground, and classify each element
                (no harness loop; pure perception scaffolding).

  composition — D drafts two SVG skeletons at different temperatures;
                E splices assets into placeholders; hybrid critic V
                (VLM + programmatic checkers) drives T≤4 refinement
                rounds with best-so-far reversion. A final visual-polish
                pass adjusts text/rect/arrow colours under a 3-VLM
                acceptance gate.

Quickstart:
    from editor_v2 import Editor
    e = Editor()
    out = e.run("input.png", out_dir="/tmp/out")
    print(out.final_svg, out.final_png, out.score)
"""
from .pipeline import Editor, RunOutputs

__version__ = "2.0.0"
__all__ = ["Editor", "RunOutputs", "__version__"]

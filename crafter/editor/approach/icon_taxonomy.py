"""Icon kind taxonomy for Approach 1.

Two categories qualify for raster preservation (gpt-image-2 + SAM3):

  A — content cannot be reconstructed as vector code at all:
      All visual icons / logos / symbols (snowflakes, clocks,
      axes, stars, plus marks, tags, etc. — anything with intentional
      iconography); photographs, filmstrips, video thumbnails, real
      images embedded in the figure.

  B — vector reconstruction would be impractically complex (would
      require many specific coordinates / colours / data values that
      a generation model cannot infer):
      rubik's cubes, 3D scatter plots, dense heatmaps, dense bar
      charts / histograms, molecular graphs, neural-network spaghetti
      diagrams, gaussian-splat point clouds, attention-matrix grids,
      letter-circle / letter-tile / button icons with internal
      composition, etc.

Default is RASTER. Only explicitly listed simple primitives — pure
flat colour blocks, single bars, single arrows, lines, generic panels
— go to the vector path. Decorative icons (snowflake, axes, star,
etc.) belong on the raster path, not the vector-code path; LLM-written
SVG cannot reproduce them faithfully.
"""

# Kinds that go to the VECTOR-code path (excluded from raster).
# Keep this list short — only kinds that are unambiguously a flat
# colour rect, a single line / arrow, or a generic container.
KIND_VECTOR = {
    # solid colour primitives — pure <rect fill="…"/> reproduces
    "color_strip", "color_sequence", "color_patch", "single_swatch",
    # single bars / arrows / connectors
    "vertical_bar", "horizontal_bar",
    "arrow", "line", "connector",
    "marker_dot",            # single dot — pure <circle/>
    # generic empty containers
    "panel", "container",
}


def classify(kind: str) -> str:
    """Return 'raster' (default) or 'vector' (explicit list only)."""
    k = (kind or "").strip().lower()
    return "vector" if k in KIND_VECTOR else "raster"

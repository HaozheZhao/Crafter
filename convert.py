#!/usr/bin/env python3
"""Convert a raster figure into an editable SVG / set of layered assets.

Standalone alias for ``crafter edit``. Pipeline: text-prompted grounding +
gpt-image-2 cleanup + element placement → editable SVG.

    export OPENROUTER_API_KEY="sk-or-..."
    export SAM3_SERVER_URL="http://host:port"      # grounding server

    python convert.py --img figure.png --out-dir editable_out/

See ``configs/default.yaml`` to change the models used.
"""
from __future__ import annotations

import sys

from crafter.editor.cli import main as _editor_main


def main(argv=None) -> int:
    return _editor_main() or 0


if __name__ == "__main__":
    raise SystemExit(main())

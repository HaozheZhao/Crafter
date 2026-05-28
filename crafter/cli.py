"""Crafter command-line entry point.

    crafter generate --caption "..." --paper-text-file paper.txt --out fig.png
    crafter edit     --img figure.png --out-dir out/

Both subcommands read the active config (``configs/default.yaml`` unless
``$CRAFTER_CONFIG`` is set) and the ``OPENROUTER_API_KEY`` environment variable.
"""
from __future__ import annotations

import sys


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    sys.argv = [f"crafter {cmd}", *rest]
    if cmd == "generate":
        from crafter.generation.cli import main as gen_main
        return gen_main()
    if cmd == "edit":
        from crafter.editor.cli import main as edit_main
        return edit_main() or 0
    print(f"unknown command: {cmd!r}\n", __doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

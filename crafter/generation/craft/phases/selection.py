"""Variant selection.

`select_best_variant`: rank the parallel-generated variants by critic
`overall` and pick the top one. Print the comparison table. Returns
the tuple consumed by the iterative refinement loop.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from crafter.generation.craft.venue_styles import VISUAL_STYLES

if TYPE_CHECKING:
    from crafter.generation.craft.session import CraftSession, CraftInput

logger = logging.getLogger(__name__)
console = Console()


def select_best_variant(variant_results: list):
    """Sort `variant_results` (mutated in place) by critic.overall desc and
    return (best_style, best_path, current_prompt, best_critique, best_vs_name)."""
    variant_results.sort(key=lambda x: x[3].overall, reverse=True)
    best_style, best_path, current_prompt, best_critique = variant_results[0]
    best_vs_name = VISUAL_STYLES.get(best_style, {}).get("name", best_style)

    table = Table(title="Variant Comparison", show_header=True)
    table.add_column("Style", style="cyan")
    table.add_column("Content", justify="center")
    table.add_column("Layout", justify="center")
    table.add_column("Aesthetic", justify="center")
    table.add_column("Overall", justify="center", style="bold")
    for vs_key, _, _, crit in variant_results:
        vs_n = VISUAL_STYLES.get(vs_key, {}).get("name", vs_key)
        overall_color = (
            "green" if crit.is_acceptable
            else "yellow" if crit.overall >= 6 else "red"
        )
        marker = " *" if vs_key == best_style else ""
        table.add_row(
            f"{vs_n}{marker}",
            f"{crit.content_accuracy:.1f}",
            f"{crit.layout_quality:.1f}",
            f"{crit.aesthetic_quality:.1f}",
            f"[{overall_color}]{crit.overall:.1f}[/{overall_color}]",
        )
    console.print(table)
    console.print(
        f"\n  Selected: [bold green]{best_vs_name}[/bold green] "
        f"(score: {best_critique.overall:.1f})"
    )

    return best_style, best_path, current_prompt, best_critique, best_vs_name

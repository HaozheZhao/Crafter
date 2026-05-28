"""SVGGenerator: generates academic diagrams as SVG code then renders to PNG.

This produces vector-quality figures with perfect text rendering, solving the
core limitation of raster image generation (blurry/garbled text).

Pipeline:
1. LLM generates SVG code describing the diagram
2. cairosvg renders SVG → PNG at high resolution
3. The SVG file is also saved for editability (can be opened in Inkscape, draw.io)

Key advantages over raster generation:
- Perfect text rendering (never garbled, always the right size)
- Vector-quality lines and shapes (crisp at any zoom)
- Editable output (SVG can be modified by any vector editor)
- Consistent styling (exact colors, exact positions)

Limitations:
- Cannot include photographic thumbnails (only shapes and text)
- Less "artistic" than AI image generation
- Complex layouts may have alignment issues
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SVGGenerator:
    """Generates academic diagrams as SVG code."""

    def __init__(self, router):
        self.router = router

    def generate(
        self,
        description: str,
        paper_context: str,
        caption: str,
        figure_type: str = "method_pipeline",
        output_dir: str = "./output",
        name: str = "diagram",
    ) -> Optional[str]:
        """Generate an SVG diagram and render to PNG.

        Args:
            description: What the figure should show.
            paper_context: Paper text for faithfulness.
            caption: Figure caption.
            figure_type: Type of figure.
            output_dir: Where to save files.
            name: Base name for output files.

        Returns:
            Path to the rendered PNG, or None on failure.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Step 1: Generate SVG code via LLM
        svg_code = self._generate_svg_code(
            description, paper_context, caption, figure_type
        )
        if not svg_code:
            return None

        # Step 2: Validate and fix SVG
        svg_code = self._fix_svg(svg_code)

        # Step 3: Save SVG
        svg_path = str(Path(output_dir) / f"{name}.svg")
        Path(svg_path).write_text(svg_code, encoding="utf-8")
        logger.info(f"SVG saved: {svg_path} ({len(svg_code)} chars)")

        # Step 4: Render to PNG
        png_path = str(Path(output_dir) / f"{name}.png")
        try:
            import cairosvg
            cairosvg.svg2png(
                bytestring=svg_code.encode("utf-8"),
                write_to=png_path,
                scale=2,  # 2x for high-res
            )
            logger.info(f"PNG rendered: {png_path}")
            return png_path
        except Exception as e:
            logger.warning(f"SVG→PNG render failed: {e}")
            # Try fallback: save SVG and use a simpler renderer
            return self._fallback_render(svg_code, png_path)

    def _generate_svg_code(
        self,
        description: str,
        paper_context: str,
        caption: str,
        figure_type: str,
    ) -> Optional[str]:
        """Use LLM to generate SVG code for the diagram."""
        prompt = self._build_svg_prompt(description, paper_context, caption, figure_type)

        # Claude is better at code generation than Gemini
        for model in ["claude-4.6-opus", "gemini-3.1-pro-preview"]:
            try:
                resp = self.router._chat(
                    [{"role": "user", "content": prompt}],
                    model=model,
                    temperature=0.3,
                    max_tokens=50000,
                )

                svg_code = self._extract_svg(resp)
                if svg_code and "<svg" in svg_code and "</svg>" in svg_code:
                    logger.info(f"SVG generated via {model}: {len(svg_code)} chars")
                    return svg_code

            except Exception as e:
                logger.warning(f"SVG generation with {model} failed: {e}")
                continue

        return None

    def _build_svg_prompt(
        self, description: str, paper_context: str, caption: str, figure_type: str
    ) -> str:
        """Build the prompt for SVG code generation."""
        return (
            f"Generate SVG code for an academic method diagram.\n\n"
            f"## Figure Caption:\n{caption}\n\n"
            f"## Description:\n{description[:3000]}\n\n"
            f"## Paper Context:\n{paper_context[:4000]}\n\n"
            f"## LAYOUT RULES (CRITICAL):\n"
            f"- Viewport: width=1400 height=600 (WIDE landscape)\n"
            f"- Use the FULL width — spread components horizontally\n"
            f"- Each box: minimum 100px wide, 50px tall\n"
            f"- Gap between boxes: at least 40px\n"
            f"- Font size: 14-16px for labels (MUST be readable)\n"
            f"- If multi-panel (a,b,c): divide width equally, each panel ~400px wide\n\n"
            f"## STYLE RULES:\n"
            f"- White background (#FFFFFF)\n"
            f"- Rounded rectangles (rx=8) with light pastel fills:\n"
            f"  Input: #D6EAF8, Process: #D5F5E3, Output: #FADBD8,\n"
            f"  Attention: #F9E79F, Loss: #E8DAEF, Frozen: #F2F3F4\n"
            f"- Thin borders: stroke='#888' stroke-width='1.5'\n"
            f"- Arrows: stroke='#666' stroke-width='1.5' marker-end='url(#arrow)'\n"
            f"- Dashed rectangles (stroke-dasharray='6,3') for logical groups\n"
            f"- Text: font-family='Arial, Helvetica, sans-serif', fill='#333'\n\n"
            f"## TEXT RULES:\n"
            f"- Labels: SHORT (1-3 words) using exact paper terminology\n"
            f"- NO descriptions inside boxes — just names\n"
            f"- NO figure title, NO figure number\n"
            f"- Include ALL key components from the paper\n\n"
            f"## Required SVG structure:\n"
            f"Start with:\n"
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1400 600" '
            f'width="1400" height="600">\n'
            f"  <defs>\n"
            f'    <marker id=\"arrow\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\"\n'
            f'      markerWidth=\"6\" markerHeight=\"6\" orient=\"auto\">\n'
            f'      <path d=\"M 0 0 L 10 5 L 0 10 z\" fill=\"#666\"/>\n'
            f"    </marker>\n"
            f"  </defs>\n"
            f'  <rect width="1400" height="600" fill="white"/>\n\n'
            f"Then add components. End with </svg>.\n\n"
            f"Return ONLY the SVG code. No markdown, no explanation."
        )

    def _extract_svg(self, response: str) -> Optional[str]:
        """Extract SVG code from LLM response."""
        text = response.strip()

        # Remove markdown code blocks
        if "```" in text:
            lines = text.split("\n")
            svg_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    svg_lines.append(line)
            if svg_lines:
                text = "\n".join(svg_lines)

        # Find the SVG element
        match = re.search(r"(<svg[\s\S]*?</svg>)", text)
        if match:
            return match.group(1)

        # If the whole response is SVG
        if text.startswith("<svg") and text.endswith("</svg>"):
            return text

        return None

    def _fix_svg(self, svg_code: str) -> str:
        """Fix common SVG issues from LLM generation."""
        # Ensure xmlns is present
        if 'xmlns=' not in svg_code:
            svg_code = svg_code.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"', 1)

        # Fix orient="auto-start-auto" (cairosvg doesn't support it)
        svg_code = svg_code.replace('orient="auto-start-auto"', 'orient="auto"')

        # Fix unescaped ampersands
        svg_code = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;|#)', '&amp;', svg_code)

        # Fix unclosed tags (common LLM error)
        svg_code = re.sub(r'<text([^>]*)/>',
                          lambda m: f'<text{m.group(1)}></text>', svg_code)

        # Remove unsupported CSS properties
        svg_code = svg_code.replace('filter: drop-shadow', '/* filter: drop-shadow')

        return svg_code

    def _fallback_render(self, svg_code: str, png_path: str) -> Optional[str]:
        """Fallback PNG rendering using svgwrite + PIL."""
        try:
            # Try rendering with a simplified approach
            import cairosvg

            # Simplify SVG if it failed
            # Remove potentially problematic elements
            simplified = re.sub(r'<image[^>]*/?>', '', svg_code)
            simplified = re.sub(r'<foreignObject[\s\S]*?</foreignObject>', '', simplified)

            cairosvg.svg2png(
                bytestring=simplified.encode("utf-8"),
                write_to=png_path,
                scale=2,
            )
            return png_path
        except Exception as e:
            logger.warning(f"Fallback render also failed: {e}")
            return None

    def generate_variants(
        self,
        description: str,
        paper_context: str,
        caption: str,
        figure_type: str = "method_pipeline",
        output_dir: str = "./output",
        num_variants: int = 2,
    ) -> list[str]:
        """Generate multiple SVG variants with different layouts.

        Returns list of PNG paths.
        """
        variants = []
        layouts = [
            ("horizontal", "Layout: LEFT-TO-RIGHT flow, all components in a single row"),
            ("vertical", "Layout: TOP-TO-BOTTOM flow, components stacked vertically"),
            ("multi_panel", "Layout: Multi-panel (a,b,c) with different aspects in each panel"),
        ]

        for i, (layout_name, layout_desc) in enumerate(layouts[:num_variants]):
            name = f"svg_variant_{layout_name}"
            desc_with_layout = f"{description}\n\n{layout_desc}"

            path = self.generate(
                description=desc_with_layout,
                paper_context=paper_context,
                caption=caption,
                figure_type=figure_type,
                output_dir=output_dir,
                name=name,
            )
            if path:
                variants.append(path)
                logger.info(f"SVG variant {i+1}/{num_variants}: {path}")

        return variants

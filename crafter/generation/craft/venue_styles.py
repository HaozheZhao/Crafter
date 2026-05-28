"""Venue-specific and visual-style knowledge base for academic figure generation.

Two orthogonal axes:
1. VENUE_STYLES — venue-specific conventions (NeurIPS, CVPR, Nature, etc.)
2. VISUAL_STYLES — visual approach/rendering type (block diagram, conceptual
   illustration, infographic, hand-drawn, etc.)

The final style prompt = venue style + visual style + figure-type hints.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Visual Styles — different rendering/artistic approaches
# ──────────────────────────────────────────────────────────────────────────────

VISUAL_STYLES: dict[str, dict] = {
    "block_diagram": {
        "name": "Block Diagram / Pipeline",
        "description": "Professional boxes-and-arrows method diagram for academic papers",
        "style_prefix": """\
Render as a professional academic BLOCK DIAGRAM:
- Rounded rectangles with soft pastel fills (light blue, light green, light orange, light purple)
- Clean layout with generous whitespace between components
- Clean gray arrows showing data flow direction
- Short labels inside boxes in sans-serif font (Arial/Helvetica style)
- Light-fill containers with dashed borders to group logical stages
- Left-to-right or top-to-bottom flow
- WHITE background
- Where relevant, include small visual elements: thumbnail images for inputs,
  colored token bars for sequences, small grids for feature maps
- Use color purposefully: warm tones for trainable, cool tones for frozen
- Consistent spacing and alignment between all components""",
    },

    "conceptual_illustration": {
        "name": "Conceptual Illustration",
        "description": "Visually rich academic diagram with visual elements and metaphors",
        "style_prefix": """\
Create a visually rich ACADEMIC DIAGRAM:
- Use visual elements to make concepts concrete: thumbnail images for inputs/outputs,
  colored grids for feature maps, small charts for results
- Professional pastel colors — not garish but not dull either
- Clear visual hierarchy with grouped sections
- Short text labels only, let visual elements tell the story
- Small inset diagrams or callouts for technical detail
- WHITE or very light background
- Can include small representative images, icons, or visual metaphors
- Clean arrows with proper routing
- This is a paper figure — professional but visually engaging""",
    },

    "infographic": {
        "name": "Infographic / Visual Summary",
        "description": "Data-driven visual summary with icons, stats, and visual hierarchy",
        "style_prefix": """\
Design as a professional INFOGRAPHIC with:
- Clear visual hierarchy: large title, section headers, body content
- Custom icons or pictograms for each concept (not generic clip-art)
- Statistics and numbers displayed prominently with large fonts
- Color-coded sections with distinct but harmonious palette
- Timeline, circular flow, or numbered steps layout
- Subtle background textures or gradients (light, not distracting)
- Mix of text, icons, small charts, and visual elements
- Professional typography: bold headers, clean body text
- Data visualization elements: mini bar charts, progress rings, comparison bars
- Think "research poster" or "one-page summary" style
- WHITE or very light colored background""",
    },

    "flowchart": {
        "name": "Flowchart / Decision Tree",
        "description": "Process flow with decision points, branches, and conditional paths",
        "style_prefix": """\
Create a structured FLOWCHART with:
- Standard flowchart shapes: rectangles for processes, diamonds for decisions, \
ovals for start/end, parallelograms for I/O
- Clean connecting arrows with labels on decision branches (yes/no, true/false)
- Color-coded paths for different execution branches
- Consistent shape sizes and spacing
- Top-to-bottom or left-to-right primary flow
- Swim lanes if multiple actors or systems are involved
- Clear entry and exit points
- Sans-serif labels inside shapes
- Light fills with darker borders for each shape
- WHITE background, professional appearance""",
    },

    "multi_panel": {
        "name": "Multi-Panel Figure",
        "description": "Nature/Science style with labeled sub-panels (a, b, c, d)",
        "style_prefix": """\
Create a MULTI-PANEL FIGURE in Nature/Science style:
- Panel layout: labeled sub-panels with bold lowercase letters (a, b, c, d)
- Each panel shows a different aspect: overview, detail, result, comparison
- Panels separated by subtle whitespace, aligned on a grid
- Muted, sophisticated color palette — no bright neon
- Thin lines, elegant typography (sans-serif)
- Scale bars and proper unit labels where applicable
- Panel (a): usually the method overview or schematic
- Panel (b-d): supporting details, results, comparisons
- Consistent styling across all panels
- Color-blind friendly palette preferred
- Vector-quality crispness
- WHITE or very light gray background""",
    },

    "comparison_grid": {
        "name": "Comparison Grid / Table Figure",
        "description": "Side-by-side comparison of methods, results, or approaches",
        "style_prefix": """\
Create a COMPARISON GRID figure:
- Organized in rows and columns with clear headers
- Each cell shows a result, example, or visualization
- Row headers: different methods or conditions
- Column headers: different metrics, datasets, or examples
- Consistent cell sizing and alignment
- Color-coded highlights for best results or key differences
- Optional: checkmarks, X marks, or colored indicators
- Borders between cells: thin, subtle gray lines
- Header row/column with slightly different background shade
- Clean typography, sans-serif fonts
- WHITE background, publication-ready layout""",
    },

    "annotated_diagram": {
        "name": "Annotated Technical Diagram",
        "description": "Detailed technical diagram with callouts, annotations, and zoomed insets",
        "style_prefix": """\
Create a detailed ANNOTATED TECHNICAL DIAGRAM:
- Central main diagram with the primary concept
- Callout boxes or zoomed inset panels showing details
- Leader lines connecting annotations to specific parts
- Mathematical notation where appropriate (loss functions, equations)
- Dimension labels and size indicators
- Color highlighting for regions of interest
- Optional magnified views of critical components
- Professional technical drawing style
- Mix of schematic and detailed views
- Clear visual hierarchy: main diagram dominant, annotations secondary
- WHITE background, thin precise lines""",
    },

    "timeline": {
        "name": "Timeline / Process Steps",
        "description": "Sequential process shown as numbered steps or timeline",
        "style_prefix": """\
Create a TIMELINE or STEP-BY-STEP PROCESS figure:
- Horizontal or vertical timeline with numbered steps
- Each step has an icon/illustration and brief description
- Connecting line or arrow between steps showing progression
- Color gradient or distinct colors per phase
- Step numbers in circles or badges
- Brief labels below/beside each step
- Optional: branching timelines for parallel processes
- Progress indicators between steps
- Clean, modern design with ample whitespace
- Icons should be meaningful, not decorative
- WHITE background, professional layout""",
    },

    "data_visualization": {
        "name": "Data Visualization / Chart",
        "description": "Charts, plots, and data-driven visualizations",
        "style_prefix": """\
Create a professional DATA VISUALIZATION:
- Clean axes with proper labels and units
- Subtle gridlines (light gray, not prominent)
- Distinct colors per data series with clear legend
- Chart types: bar chart, line plot, scatter plot, heatmap, radar chart as appropriate
- Error bars or confidence intervals where applicable
- Statistical annotations (p-values, significance markers)
- Data points clearly visible, not cluttered
- Proper aspect ratio for the data being shown
- Title and axis labels in sans-serif font
- Color-blind friendly palette
- WHITE background, publication-quality appearance""",
    },

    "equation_figure": {
        "name": "Equation / Mathematical Figure",
        "description": "Figure centered on mathematical formulations with visual explanations",
        "style_prefix": """\
Create a MATHEMATICAL FIGURE that visualizes equations and formulations:
- Key equations displayed prominently with proper mathematical typesetting
- Visual annotations explaining each term (colored underbraces, arrows to terms)
- Color-coded variables matching between equation and diagram
- Geometric or visual interpretation alongside the math
- Clean, minimal design — the math is the centerpiece
- Proper mathematical notation (subscripts, superscripts, Greek letters)
- Optional: small accompanying diagram showing what the equation describes
- Serif font for math, sans-serif for labels
- Light color backgrounds for equation boxes
- WHITE overall background""",
    },

    "research_poster": {
        "name": "Research Conference Poster",
        "description": "Academic conference poster with title banner, multi-column layout, and visual hierarchy for large-format printing",
        "style_prefix": """\
Design a professional ACADEMIC CONFERENCE POSTER:

LAYOUT STRUCTURE:
- TOP BANNER: Large bold paper title, author names, affiliations, conference logo/badge
- MAIN BODY: Multi-column layout with clear section headers (2 columns for portrait, 3 for landscape)
- BOTTOM: Key takeaway or conclusion in a highlighted strip
- Follow the orientation specified in the description (portrait or landscape)

TYPOGRAPHY (designed for large-format printing):
- Paper title: very large, bold, dominant — the first thing visible from 10 feet away
- Section headers: large, bold, colored background bars (e.g., "Motivation", "Method", "Results")
- Body text: large and readable — NO tiny text, NO dense paragraphs
- Key numbers/metrics: displayed extra-large in colored boxes or badges
- Author names: clearly listed below title with institutional affiliations

VISUAL DESIGN:
- Clean WHITE or very light background for the poster body
- Color-coded sections: each column or section has a subtle pastel background tint
- Architecture/method diagrams: clean block diagrams with labeled components
- Result visualizations: bar charts, comparison tables, or scatter plots with large labels
- Qualitative examples: small image grids showing input/output pairs
- Icons and visual elements preferred over long text descriptions
- Generous whitespace between sections — do NOT cram content
- Rounded section containers with soft shadows or borders

CONTENT BALANCE:
- Motivation column: problem statement, key challenges (use icons/bullets, not paragraphs)
- Method column: architecture diagram, training pipeline, key equations (visual, not text-heavy)
- Results column: quantitative results (large numbers), comparison chart, qualitative examples
- Each section should be ~30% visual elements, ~70% structured content (NOT paragraphs)

POSTER-SPECIFIC RULES:
- NO figure numbers or "Figure X:" labels — this IS the figure
- NO paper citations or reference lists
- Minimize text — use bullet points, icons, and diagrams instead of paragraphs
- Every section must be readable when the poster is viewed from 4-6 feet away
- Use visual hierarchy: size and color to guide the viewer's eye
- Include the conference name/year as a badge in the top-right corner""",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Venue Styles — venue-specific conventions and aesthetics
# ──────────────────────────────────────────────────────────────────────────────

VENUE_STYLES: dict[str, dict] = {
    "neurips": {
        "name": "NeurIPS / ICML",
        "style_prefix": """\
Study the reference images' visual style and match it:
- WHITE background everywhere — no black or dark areas
- Soft pastel rounded boxes with subtle shadows or borders
- Color coding: light green for encoders/feature extractors, light blue for \
language models/transformers, light orange for decoders/generators, \
light purple for special modules, light yellow for data/embeddings
- Token sequences shown as small colored squares in a row/grid
- Real example images (photos) as input/output when applicable
- Professional, clean academic publication style
- Sans-serif font (Arial/Helvetica) for all labels
- Consistent spacing between components""",
        "figure_type_hints": {
            "method_pipeline": "Left-to-right flow. Separate stages with dashed lines. Show data flow with arrows.",
            "architecture": "Top-to-bottom hierarchy. Nested containers for layers. Vertical arrows between levels.",
            "teaser": "Single wide figure showing the core idea at a glance. Central concept prominent.",
            "comparison": "Side-by-side panels. Consistent layout between compared approaches.",
            "result_chart": "Clean axes, subtle gridlines, distinct colors per series. Legend in corner.",
            "poster": "Landscape 3-column poster. Blue/teal accent colors. Clean architecture diagram in method section. Bar chart or table for results. NeurIPS-style pastel blocks.",
        },
        "search_queries": [
            "NeurIPS 2024 method figure pipeline diagram",
            "ICML 2024 architecture figure neural network",
            "NeurIPS best paper figure methodology",
        ],
        "recommended_visual_styles": ["block_diagram", "multi_panel", "conceptual_illustration", "annotated_diagram"],
    },

    "acl": {
        "name": "ACL / EMNLP / NAACL",
        "style_prefix": """\
Study the reference images' visual style and match it:
- WHITE background, clean and professional
- Rectangular boxes with colored fills for different components
- Color coding: blue tones for encoders/embeddings, green for generation, \
orange for attention/alignment, gray for frozen/pretrained modules
- Arrows with labels showing data flow
- Text-heavy figures are common — show token sequences, attention patterns
- NLP-specific: show input text → processing → output text flow
- Sans-serif fonts, clean borders
- Moderate use of color — not too flashy
- Publication-quality for ACL/EMNLP format""",
        "figure_type_hints": {
            "method_pipeline": "Show text processing pipeline. Include example text tokens.",
            "architecture": "Transformer-style stacked layers. Show attention patterns.",
            "teaser": "Example-driven: show input/output examples alongside the method.",
            "comparison": "Table-style comparison or side-by-side model outputs.",
            "poster": "Landscape 3-column poster. Show NLP pipeline with token examples. Include benchmark comparison table. ACL-style blue/white color scheme.",
        },
        "search_queries": [
            "ACL 2024 method figure NLP pipeline",
            "EMNLP 2024 transformer architecture diagram",
        ],
        "recommended_visual_styles": ["block_diagram", "annotated_diagram", "comparison_grid", "flowchart"],
    },

    "cvpr": {
        "name": "CVPR / ICCV / ECCV",
        "style_prefix": """\
Study the reference images' visual style and match it:
- WHITE background with clean layout
- Vision-focused: show image inputs/outputs prominently
- Feature maps visualized as colored grids or heatmaps
- Color coding: blue for backbone/encoder, orange/red for detection heads, \
green for segmentation, purple for feature fusion
- Clean arrows showing data flow through network
- Include example images (photos) as input/output
- Bounding boxes, masks, or keypoints on example outputs
- Professional computer vision conference style
- Moderate complexity — show key components clearly""",
        "figure_type_hints": {
            "method_pipeline": "Show image → backbone → heads → predictions flow with example images.",
            "architecture": "Feature pyramid or multi-scale architecture. Show resolution changes.",
            "teaser": "Dramatic input/output comparison. Before/after or detection results.",
            "comparison": "Visual comparison of results (ours vs baselines) on same images.",
            "poster": "Landscape 3-column poster. Include example input/output image pairs. Show architecture with feature maps. Visual comparison grid for results.",
        },
        "search_queries": [
            "CVPR 2024 method figure detection segmentation",
            "ICCV 2024 architecture diagram vision transformer",
        ],
        "recommended_visual_styles": ["block_diagram", "multi_panel", "comparison_grid", "conceptual_illustration"],
    },

    "iclr": {
        "name": "ICLR",
        "style_prefix": """\
Study the reference images' visual style and match it:
- WHITE background, clean and minimal
- Similar to NeurIPS style but often more conceptual
- Soft colors, rounded rectangles
- Mathematical notation integrated naturally
- Clean arrows and flow lines
- Emphasis on clarity over decoration
- Good balance of visual elements and whitespace
- Professional representation learning / deep learning style""",
        "figure_type_hints": {
            "method_pipeline": "Show learning pipeline with loss functions and data flow.",
            "architecture": "Model architecture with skip connections and residual paths.",
            "teaser": "Conceptual illustration of the key insight or contribution.",
            "poster": "Landscape 3-column poster. Conceptual method diagram with loss annotations. Clean result tables. ICLR-style minimal aesthetic.",
        },
        "search_queries": [
            "ICLR 2024 method figure representation learning",
            "ICLR 2025 architecture diagram",
        ],
        "recommended_visual_styles": ["block_diagram", "conceptual_illustration", "equation_figure", "annotated_diagram"],
    },

    "nature": {
        "name": "Nature / Science",
        "style_prefix": """\
Study the reference images' visual style and match it:
- WHITE or very light gray background
- Highly polished, magazine-quality aesthetics
- Multi-panel layout with labeled panels (a, b, c, d)
- Muted, sophisticated color palette — avoid bright neon colors
- Thin lines and elegant typography
- Scale bars and proper unit labels
- Nature-style: clean, authoritative, slightly conservative
- Vector-quality crispness
- Detailed but not cluttered
- Color-blind friendly palette preferred""",
        "figure_type_hints": {
            "method_pipeline": "Multi-panel figure. Panel a: overview. Panel b-d: details.",
            "architecture": "Schematic diagram with biological/chemical accuracy if applicable.",
            "teaser": "Elegant summary figure with sub-panels.",
            "result_chart": "Publication-quality plots with error bars and statistical annotations.",
            "poster": "Landscape poster with elegant multi-panel layout. Muted sophisticated colors. Nature-style polish with error bars on all quantitative results.",
        },
        "search_queries": [
            "Nature 2024 method figure AI machine learning",
            "Science 2024 schematic diagram deep learning",
        ],
        "recommended_visual_styles": ["multi_panel", "conceptual_illustration", "data_visualization", "annotated_diagram"],
    },

    "aaai": {
        "name": "AAAI",
        "style_prefix": """\
Study the reference images' visual style and match it:
- WHITE background, standard academic style
- Clear, well-organized layout
- Color-coded components with soft fills
- Arrows showing data/control flow
- Module-level abstraction
- Clean sans-serif typography
- Consistent with AI/ML conference standards""",
        "figure_type_hints": {
            "method_pipeline": "Standard pipeline diagram with clear flow.",
            "architecture": "Modular architecture with labeled components.",
            "poster": "Landscape 3-column poster. Clean modular architecture diagram. Standard AI conference style with color-coded sections.",
        },
        "search_queries": [
            "AAAI 2024 method figure pipeline architecture",
        ],
        "recommended_visual_styles": ["block_diagram", "flowchart", "annotated_diagram"],
    },

    "general": {
        "name": "General Academic",
        "style_prefix": """\
Generate a professional academic figure:
- WHITE background
- Soft pastel colored boxes (light blue, light green, light orange, light purple)
- Clean arrows connecting components
- Sans-serif font labels
- Organized left-to-right or top-to-bottom layout
- No decorative clutter
- Publication-quality appearance""",
        "figure_type_hints": {
            "method_pipeline": "Clear sequential flow with labeled stages.",
            "architecture": "Hierarchical component diagram.",
            "teaser": "High-level overview of the approach.",
            "poster": "Landscape 3-column poster layout. Professional color scheme. Clean diagrams and result visualizations.",
        },
        "search_queries": [
            "academic paper method figure diagram 2024",
        ],
        "recommended_visual_styles": ["block_diagram", "conceptual_illustration", "infographic"],
    },
}

# Aliases
VENUE_STYLES["icml"] = VENUE_STYLES["neurips"]
VENUE_STYLES["iccv"] = VENUE_STYLES["cvpr"]
VENUE_STYLES["eccv"] = VENUE_STYLES["cvpr"]
VENUE_STYLES["emnlp"] = VENUE_STYLES["acl"]
VENUE_STYLES["naacl"] = VENUE_STYLES["acl"]
VENUE_STYLES["science"] = VENUE_STYLES["nature"]


def build_style_prompt(
    venue: str,
    figure_type: str = "method_pipeline",
    visual_style: str = "",
) -> str:
    """Build a complete style prefix for generation prompts.

    Combines venue-specific conventions with the chosen visual style.

    Args:
        venue: Target venue (e.g., "neurips", "acl", "cvpr").
        figure_type: Type of figure (e.g., "method_pipeline", "architecture").
        visual_style: Visual rendering style (e.g., "block_diagram", "conceptual_illustration").
            If empty, uses "block_diagram" as default.

    Returns:
        Style prefix string to prepend to generation prompts.
    """
    venue_lower = venue.lower().strip()
    vs = VENUE_STYLES.get(venue_lower, VENUE_STYLES["general"])

    parts = [vs["style_prefix"]]

    # Add figure-type-specific hints
    hints = vs.get("figure_type_hints", {})
    type_hint = hints.get(figure_type, hints.get("method_pipeline", ""))
    if type_hint:
        parts.append(f"\nFigure type guidance: {type_hint}")

    # Add visual style instructions
    # For poster figure type, default to research_poster style instead of block_diagram
    if not visual_style:
        visual_style = "research_poster" if figure_type == "poster" else "block_diagram"
    vis = VISUAL_STYLES.get(visual_style)
    if vis:
        parts.append(f"\n## Visual Style: {vis['name']}\n{vis['style_prefix']}")

    return "\n".join(parts)


def get_recommended_visual_styles(venue: str) -> list[str]:
    """Get recommended visual styles for a venue.

    Args:
        venue: Target venue name.

    Returns:
        List of visual style keys recommended for this venue.
    """
    venue_lower = venue.lower().strip()
    vs = VENUE_STYLES.get(venue_lower, VENUE_STYLES["general"])
    return vs.get("recommended_visual_styles", list(VISUAL_STYLES.keys())[:3])


def get_search_queries(venue: str, topic: str = "") -> list[str]:
    """Get suggested search queries for finding reference figures.

    Args:
        venue: Target venue name.
        topic: Paper topic for query refinement.

    Returns:
        List of search query strings.
    """
    venue_lower = venue.lower().strip()
    style = VENUE_STYLES.get(venue_lower, VENUE_STYLES["general"])
    queries = list(style.get("search_queries", []))

    if topic:
        queries.insert(0, f"{style['name']} 2024 figure {topic}")

    return queries


def list_venues() -> list[str]:
    """List all supported venue names."""
    return sorted(set(
        k for k in VENUE_STYLES.keys()
        if not k.startswith("_")
    ))


def list_visual_styles() -> list[str]:
    """List all available visual style keys."""
    return sorted(VISUAL_STYLES.keys())

"""PaperReader: iterative, goal-directed paper understanding agent.

Works for both PDF (visual reading) and raw text input. The reader is
iterative — it makes multiple passes to ensure extraction completeness:

  Pass 1: Overview — title, abstract, method summary, figure-relevant sections
  Pass 2: Targeted — extract specific components needed for the target figure
  Pass 3: Verification — check completeness against caption, fill gaps
  Pass N: Feedback — re-read based on downstream agent requests

The reader knows what downstream agents (planner, stylist, drawer) need
and extracts information accordingly.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PaperContext:
    """Everything downstream agents need to know about the paper."""

    # Paper metadata
    title: str = ""
    abstract: str = ""

    # Method understanding (what planner needs)
    method_summary: str = ""
    components: list[dict] = field(default_factory=list)
    connections: list[dict] = field(default_factory=list)
    layout_hint: str = ""
    key_detail: str = ""

    # Visual context (what drawer needs)
    existing_figures: list[dict] = field(default_factory=list)  # {page, caption, description}
    visual_style_hints: str = ""  # inferred from paper's existing figures

    # For faithfulness verification (what critic needs)
    component_names_raw: list[str] = field(default_factory=list)  # exact names from paper
    equations: list[str] = field(default_factory=list)  # key equations
    terminology: dict = field(default_factory=dict)  # term → definition
    concrete_examples: list[str] = field(default_factory=list)  # actual data/input examples
    data_instances: list[str] = field(default_factory=list)  # specific values/tokens

    # Source tracking
    source_type: str = ""  # "pdf_visual", "text", "hybrid"
    pages_read: list[int] = field(default_factory=list)
    read_passes: int = 0
    confidence: float = 0.0

    def topology_summary(self) -> str:
        """Deterministic topology analysis on the connections graph.

        Currently unused — wiring this output into the Designer prompt
        biased the generator toward detail-rich output and regressed
        conciseness on academic T2I samples. Kept available for future
        research toggles.
        """
        if not self.connections:
            return ""
        from collections import defaultdict, Counter
        out_deg: dict[str, list[str]] = defaultdict(list)
        in_deg: dict[str, list[str]] = defaultdict(list)
        edges: list[tuple[str, str, str]] = []
        for c in self.connections:
            if not isinstance(c, dict):
                continue
            src = str(c.get("from", "")).strip()
            dst = str(c.get("to", "")).strip()
            lbl = str(c.get("label", "")).strip()
            if not src or not dst or src == dst.replace(" (loop)", ""):
                # Skip empty + simple self-loop placeholders
                pass
            out_deg[src].append(dst)
            in_deg[dst].append(src)
            edges.append((src, dst, lbl))

        # 1. Branching: node with > 1 distinct outgoing targets
        branches = [(n, list(dict.fromkeys(ts)))
                    for n, ts in out_deg.items()
                    if len(set(ts)) > 1]
        # 2. Merging: node with > 1 distinct incoming sources
        merges = [(n, list(dict.fromkeys(ss)))
                  for n, ss in in_deg.items()
                  if len(set(ss)) > 1]
        # 3. Feedback: cycle (back-edge to ancestor). DFS detection.
        feedback_edges: list[tuple[str, str]] = []
        adj = {n: list(set(ts)) for n, ts in out_deg.items()}
        WHITE, GRAY, BLACK = 0, 1, 2
        color = Counter()
        def dfs(u: str, stack: list[str]):
            color[u] = GRAY
            for v in adj.get(u, []):
                if color[v] == GRAY:
                    feedback_edges.append((u, v))
                elif color[v] == WHITE:
                    dfs(v, stack + [u])
            color[u] = BLACK
        for n in list(adj.keys()):
            if color[n] == WHITE:
                try:
                    dfs(n, [])
                except RecursionError:
                    break
        # 4. Parallel: 2+ source nodes (in_deg=0) AND 2+ sink nodes (out_deg=0),
        #    AND the source nodes feed disjoint subgraphs (rough heuristic).
        sources = [n for n in set(out_deg) | set(in_deg)
                   if not in_deg.get(n)]
        sinks = [n for n in set(out_deg) | set(in_deg)
                 if not out_deg.get(n)]
        parallel_hint = (len(sources) >= 2 and len(sinks) >= 1)

        notes: list[str] = []
        if branches:
            for src, targets in branches[:3]:
                tgts = ", ".join(targets[:4])
                notes.append(
                    f"BRANCHING: `{src}` splits into multiple downstream "
                    f"elements ({tgts}) — render as one source with "
                    f"explicit arrows to each branch, NOT a single arrow"
                )
        if merges:
            for dst, srcs in merges[:3]:
                ss = ", ".join(srcs[:4])
                notes.append(
                    f"MERGING: multiple upstream elements ({ss}) feed into "
                    f"`{dst}` — render arrows converging at `{dst}`"
                )
        if feedback_edges:
            for u, v in feedback_edges[:2]:
                notes.append(
                    f"FEEDBACK LOOP: `{u}` → `{v}` is a back-edge in the "
                    f"data flow — render with a clearly visible looping arrow"
                )
        if parallel_hint and not (branches or merges):
            # No explicit branch/merge but multiple sources/sinks suggests
            # disjoint pipelines running in parallel.
            notes.append(
                f"PARALLEL PIPELINES: {len(sources)} independent source "
                f"nodes feed into {len(sinks)} sink node(s) — the figure "
                f"may need to show 2+ parallel streams, not a single chain"
            )
        return "\n".join(f"- {n}" for n in notes)

    def for_planner(self, max_chars: int = 15000) -> str:
        """Format context for the planner agent. No aggressive truncation."""
        parts = []
        if self.title:
            parts.append(f"# {self.title}")
        if self.method_summary:
            # Use generous limit — reader already processed the long doc
            summary_limit = max(5000, max_chars - 2000)
            parts.append(f"\n## Method\n{self.method_summary[:summary_limit]}")
        if self.components:
            comp_str = "\n".join(
                f"- **{c['name']}**: {c.get('role', '')}" for c in self.components[:15]
            )
            parts.append(f"\n## Components\n{comp_str}")
        if self.connections:
            conn_str = "\n".join(
                f"- {c['from']} → {c['to']}: {c.get('label', '')}" for c in self.connections[:15]
            )
            parts.append(f"\n## Data Flow\n{conn_str}")
        if self.layout_hint:
            parts.append(f"\n## Layout: {self.layout_hint}")
        if self.key_detail:
            parts.append(f"\n## Key Insight: {self.key_detail}")
        if self.equations:
            parts.append(f"\n## Key Equations\n" + "\n".join(f"- {e}" for e in self.equations[:5]))
        if self.concrete_examples:
            parts.append(f"\n## Concrete Examples (show these in the diagram)\n" +
                        "\n".join(f"- {e}" for e in self.concrete_examples[:5]))
        if self.data_instances:
            parts.append(f"\n## Data Instances\n" +
                        "\n".join(f"- {e}" for e in self.data_instances[:5]))
        result = "\n".join(parts)
        return result[:max_chars]

    def for_critic(self) -> str:
        """Format context for the critic agent — names that must appear."""
        if self.component_names_raw:
            return "Required components: " + ", ".join(self.component_names_raw)
        if self.components:
            return "Required components: " + ", ".join(c["name"] for c in self.components)
        return ""


class PaperReader:
    """Iterative paper reading agent that knows what downstream agents need."""

    def __init__(self, router):
        self.router = router

    def read(
        self,
        caption: str,
        category: str = "",
        pdf_path: str = "",
        raw_text: str = "",
        max_passes: int = 3,
    ) -> PaperContext:
        """Iteratively read a paper to extract what downstream agents need.

        Args:
            caption: Figure caption — defines what we need to extract.
            category: Paper category (agent_reasoning, vision_perception, etc.).
            pdf_path: Path to PDF file (for visual reading).
            raw_text: Raw text content.
            max_passes: Maximum reading passes.

        Returns:
            PaperContext with structured extraction.
        """
        ctx = PaperContext(source_type="text" if not pdf_path else "hybrid")

        # Determine what we need to extract based on caption
        extraction_goals = self._analyze_caption(caption, category)
        logger.info(f"Reader goals: {extraction_goals}")

        # ── Pass 1: Overview ──
        logger.info("Reader pass 1: overview")
        if pdf_path and Path(pdf_path).exists():
            page_images = self._render_pages(pdf_path, max_pages=8)
            if page_images:
                ctx.source_type = "hybrid"
                overview = self._visual_overview(page_images, caption)
                if overview:
                    ctx.method_summary = overview.get("method_summary", "")
                    ctx.title = overview.get("title", "")
                    ctx.abstract = overview.get("abstract", "")
                    ctx.pages_read = overview.get("method_pages", [])
                    ctx.existing_figures = overview.get("figures", [])

        # Always use raw text as primary source (more complete)
        if raw_text:
            text_overview = self._text_overview(raw_text, caption, extraction_goals)
            # Merge: text overview supplements visual overview
            text_summary = text_overview.get("method_summary", "")
            if isinstance(text_summary, dict):
                text_summary = json.dumps(text_summary)
            text_summary = str(text_summary)
            if not ctx.method_summary:
                ctx.method_summary = text_summary
            elif text_summary:
                # Append text details to visual summary
                ctx.method_summary += "\n\n## Additional Details:\n" + text_summary[:2000]
            if not ctx.title:
                ctx.title = text_overview.get("title", "")

        ctx.read_passes = 1

        # ── Pass 2: Targeted extraction ──
        logger.info("Reader pass 2: targeted extraction")
        source = ctx.method_summary or raw_text[:8000]
        components = self._extract_components(source, caption, category, extraction_goals)
        if components:
            ctx.components = components.get("components", [])
            ctx.connections = components.get("connections", [])
            ctx.layout_hint = components.get("layout_hint", "")
            ctx.key_detail = components.get("key_detail", "")
            ctx.equations = components.get("equations", [])
            ctx.concrete_examples = components.get("concrete_examples", [])
            ctx.data_instances = components.get("data_instances", [])
            ctx.component_names_raw = [c["name"] for c in ctx.components]
        ctx.read_passes = 2

        # ── Pass 3: Verification — are we missing anything? ──
        if max_passes >= 3 and ctx.components:
            logger.info("Reader pass 3: verification")
            gaps = self._verify_completeness(ctx, caption, extraction_goals)

            if gaps.get("missing") and raw_text:
                logger.info(f"Filling {len(gaps['missing'])} gaps: {gaps['missing']}")
                supplement = self._targeted_read(
                    raw_text, gaps["missing"], caption
                )
                if supplement:
                    # Re-extract with richer context
                    enriched = (ctx.method_summary or "") + "\n\n" + supplement
                    components2 = self._extract_components(enriched, caption, category, extraction_goals)
                    if components2 and len(components2.get("components", [])) > len(ctx.components):
                        ctx.components = components2.get("components", ctx.components)
                        ctx.connections = components2.get("connections", ctx.connections)
                        ctx.component_names_raw = [c["name"] for c in ctx.components]
                        logger.info(f"Re-extraction: {len(ctx.components)} components (was {len(components.get('components', []))})")
            ctx.read_passes = 3

        # Set confidence
        ctx.confidence = self._estimate_confidence(ctx, caption)
        logger.info(
            f"PaperReader: {ctx.read_passes} passes, {len(ctx.components)} components, "
            f"{len(ctx.connections)} connections, confidence={ctx.confidence:.2f}"
        )
        return ctx

    def refine_from_feedback(
        self,
        ctx: PaperContext,
        feedback: str,
        raw_text: str = "",
    ) -> PaperContext:
        """Re-read based on feedback from downstream agents.

        This is the feedback loop: if the critic says components are missing,
        the reader goes back and looks for them.
        """
        logger.info(f"Reader: refining from feedback: {feedback[:100]}")

        supplement = self._targeted_read(raw_text, [feedback], "")
        if supplement:
            enriched = (ctx.method_summary or "") + "\n\n## Feedback-Driven Re-Read:\n" + supplement
            ctx.method_summary = enriched[:10000]

            # Re-extract
            components = self._extract_components(enriched, "", "", [])
            if components and components.get("components"):
                new_names = {c["name"] for c in components["components"]}
                old_names = {c["name"] for c in ctx.components}
                added = new_names - old_names
                if added:
                    ctx.components.extend([c for c in components["components"] if c["name"] in added])
                    ctx.connections.extend(components.get("connections", []))
                    ctx.component_names_raw = [c["name"] for c in ctx.components]
                    logger.info(f"Added {len(added)} new components from feedback")

        ctx.read_passes += 1
        return ctx

    # ── Internal methods ──

    def _analyze_caption(self, caption: str, category: str) -> list[str]:
        """Determine what to extract based on the figure caption."""
        goals = ["method_components", "data_flow"]

        caption_lower = caption.lower()
        if any(w in caption_lower for w in ["pipeline", "overview", "framework"]):
            goals.append("pipeline_stages")
        if any(w in caption_lower for w in ["architecture", "network", "model"]):
            goals.append("network_layers")
        if any(w in caption_lower for w in ["training", "loss", "optimization"]):
            goals.append("training_details")
        if any(w in caption_lower for w in ["example", "illustration", "demo"]):
            goals.append("examples")
        if any(w in caption_lower for w in ["comparison", "ablation", "result"]):
            goals.append("baselines")

        if "vision" in category.lower():
            goals.append("visual_elements")
        if "agent" in category.lower():
            goals.append("agent_roles")

        return goals

    def _text_overview(self, text: str, caption: str, goals: list[str]) -> dict:
        """Extract overview from raw text, handling long documents via chunking.

        For documents > 10K chars, we read in chunks and merge findings.
        This prevents losing critical details from truncation.
        """
        goals_str = ", ".join(goals)

        # For short documents, single-pass reading
        if len(text) <= 15000:
            return self._read_text_chunk(text, caption, goals_str)

        # For long documents: chunked reading with guided search
        logger.info(f"Long document ({len(text)} chars), using chunked reading")

        # Step 1: Read beginning (abstract, intro, usually has method overview)
        chunk1_result = self._read_text_chunk(text[:10000], caption, goals_str)
        title = chunk1_result.get("title", "")
        overview = chunk1_result.get("method_summary", "")

        # Step 2: Search middle for method/approach section
        mid_start = len(text) // 4
        mid_end = mid_start + 15000
        if mid_end > len(text):
            mid_end = len(text)
            mid_start = max(0, mid_end - 15000)
        chunk2_text = text[mid_start:mid_end]

        # Find method-related paragraphs
        method_keywords = ["method", "approach", "algorithm", "framework", "pipeline",
                           "architecture", "model", "training", "objective", "loss"]
        paragraphs = chunk2_text.split("\n\n")
        method_paragraphs = []
        for para in paragraphs:
            if any(kw in para.lower() for kw in method_keywords):
                method_paragraphs.append(para)

        if method_paragraphs:
            method_section = "\n\n".join(method_paragraphs)[:10000]
            chunk2_result = self._read_text_chunk(method_section, caption, goals_str)
            method_detail = chunk2_result.get("method_summary", "")
            if method_detail and len(method_detail) > len(overview):
                overview = method_detail
            elif method_detail:
                overview += "\n\n" + method_detail

        # Step 3: If document is very long, also check latter portion
        if len(text) > 30000:
            latter = text[len(text)//2 : len(text)//2 + 10000]
            latter_paras = [p for p in latter.split("\n\n")
                           if any(kw in p.lower() for kw in method_keywords)]
            if latter_paras:
                chunk3 = "\n\n".join(latter_paras)[:8000]
                chunk3_result = self._read_text_chunk(chunk3, caption, goals_str)
                extra = chunk3_result.get("method_summary", "")
                if extra:
                    overview += "\n\n" + extra

        return {"title": title, "method_summary": overview}

    def _read_text_chunk(self, text: str, caption: str, goals_str: str) -> dict:
        """Read a single text chunk and extract information."""
        prompt = (
            f"Read this paper text and extract information needed for creating "
            f"a figure with caption: '{caption}'\n\n"
            f"Focus on extracting: {goals_str}\n\n"
            f"Paper text:\n{text}\n\n"
            f"Return JSON:\n"
            f'{{"title": "paper title", "method_summary": "detailed method description '
            f'covering all components, their roles, and how they connect"}}'
        )
        try:
            resp = self.router.plan(
                [{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=4000,
            )
            resp_text = resp.strip()
            if resp_text.startswith("```"):
                lines = resp_text.split("\n")
                resp_text = "\n".join(lines[1:])
                if resp_text.endswith("```"):
                    resp_text = resp_text[:-3]
            return json.loads(resp_text)
        except Exception as e:
            logger.warning(f"Text chunk reading failed: {e}")
            return {"method_summary": text[:3000]}

    def _extract_components(
        self, text: str, caption: str, category: str, goals: list[str]
    ) -> Optional[dict]:
        """Structured extraction of method components.

        Uses up to 15K chars of context for long documents.
        """
        goals_str = ", ".join(goals) if goals else "method components and data flow"
        # Use more context for extraction (15K instead of 8K)
        context = text[:15000] if len(text) > 15000 else text
        prompt = (
            f"Extract the EXACT components for a diagram from this paper.\n\n"
            f"## Figure Caption:\n{caption}\n\n"
            f"## Extraction Goals: {goals_str}\n\n"
            f"## Paper Content:\n{context}\n\n"
            f"Return JSON:\n"
            f'{{"components": [{{"name": "EXACT name from paper", "role": "5-word description"}}],\n'
            f'"connections": [{{"from": "A", "to": "B", "label": "what flows"}}],\n'
            f'"layout_hint": "left-to-right / top-down / cyclic / multi-panel",\n'
            f'"key_detail": "most important thing the figure must show",\n'
            f'"equations": ["key equation 1", "key equation 2"],\n'
            f'"concrete_examples": ["actual input/output example from the paper that should appear in the diagram"],\n'
            f'"data_instances": ["specific data shown in the figure: token sequences, actual values, sample inputs"]}}\n\n'
            f"CRITICAL:\n"
            f"1. Use EXACT terminology from the paper. Max 12 components.\n"
            f"2. If the caption mentions specific examples or data, extract them as concrete_examples.\n"
            f"3. Include any specific notation, variable names, or formulas that the figure should show."
        )
        try:
            resp = self.router.plan(
                [{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=2000,
            )
            text = resp.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:])
                if text.endswith("```"):
                    text = text[:-3]
            return json.loads(text)
        except Exception as e:
            logger.warning(f"Component extraction failed: {e}")
            return None

    def _verify_completeness(self, ctx: PaperContext, caption: str, goals: list[str]) -> dict:
        """Verify that extraction covers what the caption needs."""
        prompt = (
            f"Check if this extraction is complete for creating: '{caption}'\n\n"
            f"Components found: {json.dumps([c['name'] for c in ctx.components])}\n"
            f"Connections: {len(ctx.connections)}\n"
            f"Goals: {', '.join(goals)}\n\n"
            f"Return JSON: {{\"missing\": [\"specific info that seems missing\"], \"complete\": true/false}}"
        )
        try:
            resp = self.router._chat(
                [{"role": "user", "content": prompt}],
                model=self.router.config.quick_model,
                temperature=0.2, max_tokens=512,
            )
            text = resp.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:])
                if text.endswith("```"):
                    text = text[:-3]
            return json.loads(text)
        except Exception:
            return {"missing": [], "complete": True}

    def _targeted_read(self, text: str, missing: list[str], caption: str) -> str:
        """Search the FULL paper text for specific missing information.

        For long documents, searches across multiple chunks to find relevant passages.
        """
        missing_str = "\n".join(f"- {m}" for m in missing[:5])

        # For long documents, search across chunks
        if len(text) > 15000:
            # Search in keyword-relevant paragraphs across the entire document
            search_terms = []
            for m in missing[:5]:
                search_terms.extend(m.lower().split()[:3])

            paragraphs = text.split("\n\n")
            relevant = []
            for para in paragraphs:
                if any(term in para.lower() for term in search_terms if len(term) > 3):
                    relevant.append(para)

            if relevant:
                search_context = "\n\n".join(relevant)[:12000]
            else:
                # Fall back to middle of document (often contains method details)
                mid = len(text) // 3
                search_context = text[mid:mid + 12000]
        else:
            search_context = text

        prompt = (
            f"Search this paper for the following missing information:\n{missing_str}\n\n"
            f"Paper text:\n{search_context}\n\n"
            f"Extract and return ONLY the relevant passages."
        )
        try:
            resp = self.router._chat(
                [{"role": "user", "content": prompt}],
                model=self.router.config.quick_model,
                temperature=0.2, max_tokens=2000,
            )
            return resp
        except Exception:
            return ""

    def _render_pages(self, pdf_path: str, max_pages: int = 8) -> list[str]:
        """Render PDF pages as base64 images."""
        try:
            import pymupdf
            doc = pymupdf.open(pdf_path)
            pages = []
            for i, page in enumerate(doc):
                if i >= max_pages:
                    break
                pix = page.get_pixmap(dpi=150)
                pages.append(base64.b64encode(pix.tobytes("png")).decode())
            doc.close()
            return pages
        except Exception as e:
            logger.warning(f"PDF rendering failed: {e}")
            return []

    def _visual_overview(self, page_images: list[str], caption: str) -> Optional[dict]:
        """VLM-based visual reading of PDF pages."""
        content = []
        for idx, img in enumerate(page_images[:4]):
            content.append({"type": "text", "text": f"Page {idx+1}:"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})

        content.append({"type": "text", "text": (
            f"Read these pages. Extract information for a figure: '{caption}'\n"
            f"Return JSON: {{\"title\": \"...\", \"method_summary\": \"detailed method\", "
            f"\"method_pages\": [page numbers], \"figures\": [{{\"page\": N, \"caption\": \"...\"}}]}}"
        )})

        try:
            resp = self.router._chat(
                [{"role": "user", "content": content}],
                model=self.router.config.critic_model,
                temperature=0.3, max_tokens=3000,
            )
            text = resp.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:])
                if text.endswith("```"):
                    text = text[:-3]
            return json.loads(text)
        except Exception as e:
            logger.warning(f"Visual overview failed: {e}")
            return None

    def _estimate_confidence(self, ctx: PaperContext, caption: str) -> float:
        """Estimate confidence in the extraction."""
        score = 0.3  # base

        if ctx.components:
            score += min(0.3, len(ctx.components) * 0.03)
        if ctx.connections:
            score += min(0.15, len(ctx.connections) * 0.02)
        if ctx.method_summary and len(ctx.method_summary) > 500:
            score += 0.1
        if ctx.key_detail:
            score += 0.05
        if ctx.source_type == "hybrid":
            score += 0.1

        return min(1.0, score)

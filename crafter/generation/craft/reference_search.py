"""ReferenceSearcher: finds reference figures via Serper (Google) image search.

Searches for example academic figures that match the target venue and
figure type, downloads them, and caches locally for use as style references.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class ReferenceImage:
    """A downloaded reference image."""

    path: str = ""
    url: str = ""
    description: str = ""
    source: str = ""  # "serper", "local"


class ReferenceSearcher:
    """Searches the web for reference figures using Serper (Google Search) API."""

    SERPER_IMAGE_URL = "https://google.serper.dev/images"
    SERPER_SEARCH_URL = "https://google.serper.dev/search"

    def __init__(self, serper_api_key: str = "", cache_dir: str = ".craft_cache/references"):
        self.serper_api_key = serper_api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def search(
        self,
        topic: str,
        venue: str = "neurips",
        figure_type: str = "method_pipeline",
        max_results: int = 3,
    ) -> list[ReferenceImage]:
        """Search for reference figures matching venue and topic.

        Args:
            topic: Paper topic or keywords.
            venue: Target venue name.
            figure_type: Type of figure to find.
            max_results: Maximum number of images to return.

        Returns:
            List of ReferenceImage with cached local paths.
        """
        if not self.serper_api_key:
            logger.info("No Serper API key, skipping reference search")
            return []

        type_label = figure_type.replace("_", " ")
        query = f"{venue} 2024 {type_label} figure diagram {topic} academic paper"

        logger.info(f"Searching for references: {query}")

        try:
            response = requests.post(
                self.SERPER_IMAGE_URL,
                json={"q": query, "num": max_results * 3},
                headers={
                    "X-API-KEY": self.serper_api_key,
                    "Content-Type": "application/json",
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            image_results = data.get("images", [])
            logger.info(f"Serper returned {len(image_results)} image results")

            # Download and cache images
            references = []
            for item in image_results:
                url = item.get("imageUrl", "")
                title = item.get("title", "")
                if not url or not self._looks_like_figure_url(url):
                    continue

                ref = self._download_and_cache(url, topic, title)
                if ref:
                    references.append(ref)
                    if len(references) >= max_results:
                        break

            logger.info(f"Cached {len(references)} reference images")
            return references

        except Exception as e:
            logger.warning(f"Reference search failed: {e}")
            return []

    def search_visual_style(
        self,
        visual_style: str,
        venue: str = "neurips",
        topic: str = "",
        max_results: int = 2,
    ) -> list[ReferenceImage]:
        """Search for reference images matching a specific visual style.

        Args:
            visual_style: Visual style key (e.g., "conceptual_illustration").
            venue: Target venue name.
            topic: Paper topic.
            max_results: Maximum results to return.

        Returns:
            List of ReferenceImage with cached local paths.
        """
        if not self.serper_api_key:
            return []

        # Build style-specific queries
        style_queries = {
            "block_diagram": f"{venue} paper method pipeline block diagram figure",
            "conceptual_illustration": f"{venue} paper conceptual illustration figure artistic",
            "infographic": f"scientific infographic {topic} research visual summary",
            "flowchart": f"{venue} paper flowchart algorithm process figure",
            "multi_panel": f"Nature Science multi-panel figure {topic} labeled panels",
            "comparison_grid": f"{venue} paper comparison table figure visual grid",
            "annotated_diagram": f"{venue} paper annotated diagram callout zoom detail",
            "timeline": f"research timeline process steps figure {topic} academic",
            "data_visualization": f"{venue} paper chart plot data visualization results",
            "equation_figure": f"{venue} paper equation mathematical figure visual",
        }
        query = style_queries.get(visual_style, f"{venue} {visual_style} figure academic paper")
        if topic:
            query += f" {topic[:50]}"

        return self.search(topic=topic, venue=venue, figure_type=visual_style, max_results=max_results)

    def load_local_references(self, paths: list[str]) -> list[ReferenceImage]:
        """Load user-provided local reference images.

        Args:
            paths: List of file paths to reference images.

        Returns:
            List of ReferenceImage for valid paths.
        """
        references = []
        for path in paths:
            if Path(path).exists() and self._is_valid_image(path):
                references.append(ReferenceImage(
                    path=path,
                    url="",
                    description=f"User-provided: {Path(path).name}",
                    source="local",
                ))
            else:
                logger.warning(f"Reference image not found or invalid: {path}")
        return references

    def _looks_like_figure_url(self, url: str) -> bool:
        """Quick heuristic: does this URL look like an academic figure?"""
        url_lower = url.lower()
        skip_patterns = [
            "favicon", "icon", "avatar", "logo", "badge",
            "button", "banner", "ad", "pixel", "tracker",
            "1x1", "gravatar", "emoji", "thumbnail",
        ]
        return not any(p in url_lower for p in skip_patterns)

    def _download_and_cache(
        self, url: str, topic: str = "", title: str = ""
    ) -> Optional[ReferenceImage]:
        """Download an image URL and cache locally."""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        suffix = Path(url.split("?")[0]).suffix or ".png"
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            suffix = ".png"
        cache_path = self.cache_dir / f"ref_{url_hash}{suffix}"

        # Return cached version if exists
        if cache_path.exists():
            logger.debug(f"Cache hit: {cache_path}")
            return ReferenceImage(
                path=str(cache_path),
                url=url,
                description=title or f"Cached reference for '{topic}'",
                source="serper",
            )

        # Download
        try:
            resp = requests.get(url, timeout=15, headers={
                "User-Agent": "Mozilla/5.0 (research; academic figure reference)",
            })
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "image" not in content_type and len(resp.content) < 1000:
                return None

            cache_path.write_bytes(resp.content)

            if not self._is_valid_image(str(cache_path)):
                cache_path.unlink(missing_ok=True)
                return None

            logger.info(f"Downloaded reference: {url[:80]} -> {cache_path}")
            return ReferenceImage(
                path=str(cache_path),
                url=url,
                description=title or f"Reference for '{topic}'",
                source="serper",
            )

        except Exception as e:
            logger.debug(f"Failed to download {url}: {e}")
            return None

    @staticmethod
    def _is_valid_image(path: str) -> bool:
        """Check if a file is a valid image."""
        try:
            from PIL import Image
            img = Image.open(path)
            img.verify()
            img = Image.open(path)
            w, h = img.size
            return w >= 200 and h >= 150
        except Exception:
            return False

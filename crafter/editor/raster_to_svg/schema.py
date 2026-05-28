"""Core data types for the R2E pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ElementType(str, Enum):
    BOX = "box"
    TEXT = "text"
    ARROW = "arrow"
    ICON = "icon"
    BACKGROUND = "background"


class ShapeType(str, Enum):
    RECT = "rect"
    ROUNDED_RECT = "rounded_rect"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    DIAMOND = "diamond"
    HEXAGON = "hexagon"
    TRIANGLE = "triangle"
    CYLINDER = "cylinder"
    PARALLELOGRAM = "parallelogram"
    COMPLEX = "complex"  # cannot vectorize, keep as raster


@dataclass
class GroundedElement:
    """A detected visual element with bbox and style."""
    id: str
    element_type: ElementType
    bbox: list[int]  # [x1, y1, x2, y2] pixels
    # Style
    fill_color: str = ""
    border_color: str = ""
    border_width: int = 1
    corner_radius: int = 0
    shadow: bool = False
    opacity: float = 1.0
    # Text-specific
    text_content: str = ""
    font_size: int = 12
    font_weight: str = "normal"
    font_color: str = "#000000"
    is_equation: bool = False
    # Arrow-specific
    arrow_type: str = ""  # simple, curved, dashed
    arrow_start: list[int] = field(default_factory=list)
    arrow_end: list[int] = field(default_factory=list)
    is_code_generable: bool = True
    # Meta
    confidence: float = 0.0
    description: str = ""

    @property
    def x1(self) -> int: return self.bbox[0]
    @property
    def y1(self) -> int: return self.bbox[1]
    @property
    def x2(self) -> int: return self.bbox[2]
    @property
    def y2(self) -> int: return self.bbox[3]
    @property
    def w(self) -> int: return self.bbox[2] - self.bbox[0]
    @property
    def h(self) -> int: return self.bbox[3] - self.bbox[1]
    @property
    def area(self) -> int: return self.w * self.h


@dataclass
class GroundingResult:
    elements: list[GroundedElement] = field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)
    background_color: str = "#FFFFFF"
    layout_type: str = ""

    @property
    def icons(self) -> list[GroundedElement]:
        return [e for e in self.elements if e.element_type == ElementType.ICON]

    @property
    def boxes(self) -> list[GroundedElement]:
        return [e for e in self.elements if e.element_type == ElementType.BOX]

    @property
    def texts(self) -> list[GroundedElement]:
        return [e for e in self.elements if e.element_type == ElementType.TEXT]

    @property
    def arrows(self) -> list[GroundedElement]:
        return [e for e in self.elements if e.element_type == ElementType.ARROW]


@dataclass
class MaskQualityReport:
    is_clean: bool = False
    jagged_score: float = 0.0
    fragment_count: int = 0
    largest_fragment_ratio: float = 0.0
    boundary_smoothness: float = 0.0
    completeness: float = 0.0
    issues: list[str] = field(default_factory=list)


@dataclass
class SegmentedIcon:
    id: str
    bbox: list[int]
    mask_path: str = ""
    crop_path: str = ""
    b64: str = ""  # base64 transparent PNG
    quality: Optional[MaskQualityReport] = None
    verified: bool = False


@dataclass
class SegmentationResult:
    icons: list[SegmentedIcon] = field(default_factory=list)
    decorative_arrows: list[SegmentedIcon] = field(default_factory=list)


@dataclass
class StyledText:
    id: str
    content: str
    bbox: list[int]
    font_size: int = 12
    font_weight: str = "normal"
    font_color: str = "#000000"
    font_family: str = "sans-serif"
    alignment: str = "left"
    is_equation: bool = False
    parent_box_id: str = ""


@dataclass
class TextResult:
    texts: list[StyledText] = field(default_factory=list)


@dataclass
class VectorizedShape:
    id: str
    shape_type: ShapeType
    bbox: list[int]
    fill_color: str = "#FFFFFF"
    stroke_color: str = "#000000"
    stroke_width: int = 1
    corner_radius: int = 0
    shadow: bool = False
    text: str = ""  # label inside shape
    gradient_end_color: str = ""  # if set, shape has top-to-bottom gradient


@dataclass
class VectorizationResult:
    vectorized: list[VectorizedShape] = field(default_factory=list)
    raster_ids: list[str] = field(default_factory=list)  # element IDs kept as raster


@dataclass
class JudgeScore:
    model: str = ""
    overall: float = 0.0
    position: float = 0.0
    color: float = 0.0
    text: float = 0.0
    icon: float = 0.0
    arrow: float = 0.0
    style: float = 0.0
    issues: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class JudgeVerdict:
    passed: bool = False
    avg_score: float = 0.0
    scores: list[JudgeScore] = field(default_factory=list)
    all_issues: list[str] = field(default_factory=list)
    layout_issues: list[str] = field(default_factory=list)
    text_issues: list[str] = field(default_factory=list)
    icon_issues: list[str] = field(default_factory=list)
    grounding_issues: list[str] = field(default_factory=list)


@dataclass
class PipelineState:
    """Checkpoint state for stop/resume."""
    image_path: str = ""
    output_dir: str = ""
    round_num: int = 0
    grounding: Optional[GroundingResult] = None
    segmentation: Optional[SegmentationResult] = None
    text: Optional[TextResult] = None
    vectorization: Optional[VectorizationResult] = None
    svg_path: str = ""
    drawio_path: str = ""
    preview_path: str = ""
    mpl_preview_path: str = ""
    ground_truth: Optional[dict] = None
    judge_history: list[JudgeVerdict] = field(default_factory=list)

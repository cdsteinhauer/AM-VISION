from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ToolType = Literal["rectangle", "circle", "edge", "edge_1", "edge_2", "ai_classifier"]


@dataclass
class InspectionTool:
    id: str
    name: str
    type: ToolType = "rectangle"
    roi: tuple[float, float, float, float] = (0.2, 0.2, 0.6, 0.6)
    enabled: bool = True
    min_width_mm: float | None = None
    max_width_mm: float | None = None
    min_height_mm: float | None = None
    max_height_mm: float | None = None
    min_diameter_mm: float | None = None
    max_diameter_mm: float | None = None
    min_edge_score: float = 25.0
    min_length_mm: float | None = None
    max_length_mm: float | None = None
    line_orientation: Literal["auto", "horizontal", "vertical"] = "auto"
    debug: bool = False
    live_lines: bool = False
    min_line_length_ratio: float = 0.15
    model_dir: str = "data/models/pass_fail_classifier"
    min_confidence: float = 0.8
    min_pass_margin: float = 0.0

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "InspectionTool":
        roi = data.get("roi", [0.2, 0.2, 0.6, 0.6])
        return InspectionTool(
            id=str(data.get("id", "tool")),
            name=str(data.get("name", data.get("id", "Tool"))),
            type=_normalize_tool_type(data.get("type", "rectangle")),
            roi=tuple(float(v) for v in roi),
            enabled=bool(data.get("enabled", True)),
            min_width_mm=_optional_float(data.get("min_width_mm")),
            max_width_mm=_optional_float(data.get("max_width_mm")),
            min_height_mm=_optional_float(data.get("min_height_mm")),
            max_height_mm=_optional_float(data.get("max_height_mm")),
            min_diameter_mm=_optional_float(data.get("min_diameter_mm")),
            max_diameter_mm=_optional_float(data.get("max_diameter_mm")),
            min_edge_score=float(data.get("min_edge_score", 25.0)),
            min_length_mm=_optional_float(data.get("min_length_mm")),
            max_length_mm=_optional_float(data.get("max_length_mm")),
            line_orientation=data.get("line_orientation", "auto"),
            debug=bool(data.get("debug", False)),
            live_lines=bool(data.get("live_lines", False)),
            min_line_length_ratio=float(data.get("min_line_length_ratio", 0.15)),
            model_dir=str(data.get("model_dir", "data/models/pass_fail_classifier")),
            min_confidence=_confidence_float(data.get("min_confidence", 0.8)),
            min_pass_margin=_confidence_float(data.get("min_pass_margin", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "roi": list(self.roi),
            "enabled": self.enabled,
            "min_width_mm": self.min_width_mm,
            "max_width_mm": self.max_width_mm,
            "min_height_mm": self.min_height_mm,
            "max_height_mm": self.max_height_mm,
            "min_diameter_mm": self.min_diameter_mm,
            "max_diameter_mm": self.max_diameter_mm,
            "min_edge_score": self.min_edge_score,
            "min_length_mm": self.min_length_mm,
            "max_length_mm": self.max_length_mm,
            "line_orientation": self.line_orientation,
            "debug": self.debug,
            "live_lines": self.live_lines,
            "min_line_length_ratio": self.min_line_length_ratio,
            "model_dir": self.model_dir,
            "min_confidence": self.min_confidence,
            "min_pass_margin": self.min_pass_margin,
        }


@dataclass
class InspectionRecipe:
    name: str = "default"
    description: str = "Default rectangular part inspection"
    tools: list[InspectionTool] = field(default_factory=list)

    @staticmethod
    def default() -> "InspectionRecipe":
        return InspectionRecipe(
            tools=[
                InspectionTool(
                    id="part_rect",
                    name="Part rectangle",
                    type="rectangle",
                    roi=(0.15, 0.15, 0.7, 0.7),
                    min_width_mm=20.0,
                    max_width_mm=1000.0,
                    min_height_mm=20.0,
                    max_height_mm=1000.0,
                )
            ]
        )

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "InspectionRecipe":
        return InspectionRecipe(
            name=str(data.get("name", "default")),
            description=str(data.get("description", "")),
            tools=[InspectionTool.from_dict(item) for item in data.get("tools", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tools": [tool.to_dict() for tool in self.tools],
        }


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _confidence_float(value: Any) -> float:
    confidence = float(value)
    if confidence > 1.0:
        confidence = confidence / 100.0
    return max(0.0, min(1.0, confidence))


def _normalize_tool_type(value: Any) -> ToolType:
    raw = str(value or "rectangle")
    if raw == "edge":
        return "edge_2"
    if raw in {"rectangle", "circle", "edge_1", "edge_2", "ai_classifier"}:
        return raw
    return "rectangle"

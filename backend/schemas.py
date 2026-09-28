from pydantic import BaseModel, Field


class LesionMetrics(BaseModel):
    lesion_detected: bool
    area_px2: float
    perimeter_px: float
    equivalent_diameter_px: float
    circularity: float
    bbox_width_px: int
    bbox_height_px: int

    area_mm2: float | None = None
    perimeter_mm: float | None = None
    equivalent_diameter_mm: float | None = None
    bbox_width_mm: float | None = None
    bbox_height_mm: float | None = None


class AnalysisResponse(BaseModel):
    case_id: str
    inference_mode: str
    original_shape: list[int]
    lesion_pixel_fraction: float = Field(ge=0.0, le=1.0)
    mean_lesion_probability: float = Field(ge=0.0, le=1.0)
    metrics: LesionMetrics
    message: str

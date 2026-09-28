from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.config import get_settings
from backend.schemas import AnalysisResponse
from backend.services.analysis import (
    calculate_metrics,
    draw_overlay,
    mask_visual,
    probability_heatmap,
)
from backend.services.image_io import decode_upload, save_png
from backend.services.inference import segment


settings = get_settings()

app = FastAPI(
    title="SonoLesion-AI API",
    version="1.0.0",
    description=(
        "Research prototype for breast-ultrasound lesion segmentation, "
        "visualization, and quantitative morphology analysis."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {
        "name": "SonoLesion-AI API",
        "docs": "/docs",
        "health": "/health",
        "research_use_only": True,
    }


@app.get("/health")
def health() -> dict:
    checkpoint_ready = bool(
        settings.model_path and Path(settings.model_path).is_file()
    )
    return {
        "status": "ok",
        "model_checkpoint_configured": checkpoint_ready,
        "demo_mode_allowed": settings.allow_demo_mode,
    }


def _case_dir(case_id: str) -> Path:
    if not case_id or any(ch not in "0123456789abcdef-" for ch in case_id.lower()):
        raise HTTPException(status_code=400, detail="Invalid case ID.")

    path = settings.results_path / case_id
    if not path.is_dir():
        raise HTTPException(status_code=404, detail="Case not found.")
    return path


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_ultrasound(
    image: Annotated[UploadFile, File(description="Breast ultrasound image")],
    pixel_spacing_mm: Annotated[
        float | None,
        Form(description="Optional pixel spacing in mm/pixel"),
    ] = None,
) -> AnalysisResponse:
    if pixel_spacing_mm is not None and pixel_spacing_mm <= 0:
        raise HTTPException(
            status_code=400,
            detail="pixel_spacing_mm must be greater than zero.",
        )

    case_id = str(uuid.uuid4())
    case_dir = settings.results_path / case_id
    case_dir.mkdir(parents=True, exist_ok=False)

    try:
        _raw, image_bgr = await decode_upload(image)
        save_png(case_dir / "original.png", image_bgr)

        mask, probability, mode = segment(image_bgr)

        if mask.shape != image_bgr.shape[:2]:
            raise RuntimeError(
                f"Mask shape {mask.shape} does not match image shape {image_bgr.shape[:2]}."
            )
        if probability.shape != mask.shape:
            raise RuntimeError(
                f"Probability shape {probability.shape} does not match mask shape {mask.shape}."
            )

        metrics = calculate_metrics(mask, pixel_spacing_mm=pixel_spacing_mm)
        overlay = draw_overlay(image_bgr, mask)
        mask_img = mask_visual(mask)
        heatmap = probability_heatmap(image_bgr, probability)

        save_png(case_dir / "mask.png", mask_img)
        save_png(case_dir / "overlay.png", overlay)
        save_png(case_dir / "heatmap.png", heatmap)

        probability_path = case_dir / "probability.npy"
        np.save(probability_path, probability.astype(np.float32))

        lesion_fraction = float(np.mean(mask > 0))
        lesion_probs = probability[mask > 0]
        mean_probability = (
            float(lesion_probs.mean()) if lesion_probs.size else 0.0
        )

        metadata = {
            "case_id": case_id,
            "inference_mode": mode,
            "original_shape": list(image_bgr.shape[:2]),
            "lesion_pixel_fraction": lesion_fraction,
            "mean_lesion_probability": mean_probability,
            "pixel_spacing_mm": pixel_spacing_mm,
            "metrics": metrics.model_dump(),
        }
        (case_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        return AnalysisResponse(
            case_id=case_id,
            inference_mode=mode,
            original_shape=list(image_bgr.shape[:2]),
            lesion_pixel_fraction=round(lesion_fraction, 6),
            mean_lesion_probability=round(mean_probability, 6),
            metrics=metrics,
            message=(
                "Analysis complete. "
                + (
                    "A configured checkpoint was used."
                    if mode == "checkpoint"
                    else "Demo heuristic mode was used; output is not a learned prediction."
                )
            ),
        )
    except HTTPException:
        shutil.rmtree(case_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(case_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/v1/cases/{case_id}")
def case_summary(case_id: str) -> dict:
    case_dir = _case_dir(case_id)
    return json.loads(
        (case_dir / "metadata.json").read_text(encoding="utf-8")
    )


@app.get("/api/v1/cases/{case_id}/image/{kind}")
def case_image(case_id: str, kind: str) -> FileResponse:
    case_dir = _case_dir(case_id)
    allowed = {"original", "mask", "overlay", "heatmap"}
    if kind not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"kind must be one of {sorted(allowed)}",
        )

    path = case_dir / f"{kind}.png"
    return FileResponse(path=path, media_type="image/png")


@app.get("/api/v1/cases/{case_id}/download/{kind}")
def download_result(case_id: str, kind: str) -> FileResponse:
    case_dir = _case_dir(case_id)
    allowed = {"mask", "overlay", "heatmap"}
    if kind not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"kind must be one of {sorted(allowed)}",
        )

    path = case_dir / f"{kind}.png"
    return FileResponse(
        path=path,
        media_type="image/png",
        filename=f"{case_id}_{kind}.png",
    )

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile


ALLOWED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def valid_image_name(filename: str | None) -> bool:
    if not filename:
        return False
    return filename.lower().endswith(ALLOWED_EXTENSIONS)


async def decode_upload(upload: UploadFile) -> tuple[bytes, np.ndarray]:
    if not valid_image_name(upload.filename):
        raise HTTPException(
            status_code=400,
            detail="Supported formats: PNG, JPG, JPEG, BMP, TIF, TIFF.",
        )

    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    encoded = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image.")

    return raw, image


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Could not write image to {path}")


def grayscale(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

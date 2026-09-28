from __future__ import annotations

import math

import cv2
import numpy as np

from backend.schemas import LesionMetrics


def largest_contour(mask: np.ndarray):
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def calculate_metrics(
    mask: np.ndarray,
    pixel_spacing_mm: float | None = None,
) -> LesionMetrics:
    contour = largest_contour(mask)

    if contour is None:
        return LesionMetrics(
            lesion_detected=False,
            area_px2=0.0,
            perimeter_px=0.0,
            equivalent_diameter_px=0.0,
            circularity=0.0,
            bbox_width_px=0,
            bbox_height_px=0,
        )

    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, w, h = cv2.boundingRect(contour)

    equivalent_diameter = math.sqrt(4.0 * area / math.pi) if area > 0 else 0.0
    circularity = (
        4.0 * math.pi * area / (perimeter ** 2)
        if perimeter > 1e-8
        else 0.0
    )

    payload = dict(
        lesion_detected=area > 0,
        area_px2=round(area, 3),
        perimeter_px=round(perimeter, 3),
        equivalent_diameter_px=round(equivalent_diameter, 3),
        circularity=round(float(circularity), 4),
        bbox_width_px=int(w),
        bbox_height_px=int(h),
    )

    if pixel_spacing_mm is not None and pixel_spacing_mm > 0:
        payload.update(
            area_mm2=round(area * pixel_spacing_mm ** 2, 3),
            perimeter_mm=round(perimeter * pixel_spacing_mm, 3),
            equivalent_diameter_mm=round(
                equivalent_diameter * pixel_spacing_mm, 3
            ),
            bbox_width_mm=round(w * pixel_spacing_mm, 3),
            bbox_height_mm=round(h * pixel_spacing_mm, 3),
        )

    return LesionMetrics(**payload)


def draw_overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    contour = largest_contour(mask)

    if contour is not None:
        cv2.drawContours(overlay, [contour], -1, (0, 255, 255), 2)

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 180, 0), 2)

        area = cv2.contourArea(contour)
        cv2.putText(
            overlay,
            f"Lesion area: {area:.0f} px^2",
            (max(x, 8), max(y - 10, 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return overlay


def mask_visual(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8) * 255


def probability_heatmap(
    image_bgr: np.ndarray,
    probability: np.ndarray,
) -> np.ndarray:
    probability_u8 = np.clip(probability * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(probability_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 0.55, heatmap, 0.45, 0.0)

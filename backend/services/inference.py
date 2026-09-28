from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from monai.networks.nets import UNet

from backend.config import get_settings
from backend.services.image_io import grayscale


def build_model() -> torch.nn.Module:
    """Default 2-D MONAI U-Net adapter. Replace with EALG-Net if using your trained model."""
    return UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
    )


def preprocess_for_model(image_bgr: np.ndarray, size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    gray = grayscale(image_bgr)
    original_shape = gray.shape[:2]

    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    resized = resized.astype(np.float32) / 255.0

    mean = float(resized.mean())
    std = float(resized.std())
    if std < 1e-6:
        std = 1.0
    normalized = (resized - mean) / std

    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    return tensor, original_shape


def _clean_state_dict(state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if isinstance(state, dict):
        cleaned = {}
        for key, value in state.items():
            new_key = key
            for prefix in ("model.", "network.", "module."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            cleaned[new_key] = value
        return cleaned

    return state


def predict_with_checkpoint(
    image_bgr: np.ndarray,
    checkpoint: Path,
) -> tuple[np.ndarray, np.ndarray]:
    settings = get_settings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model().to(device)
    state = torch.load(checkpoint, map_location=device)
    state = _clean_state_dict(state)
    model.load_state_dict(state, strict=True)
    model.eval()

    tensor, original_shape = preprocess_for_model(
        image_bgr,
        settings.model_input_size,
    )
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if isinstance(logits, dict):
            logits = next(iter(logits.values()))

        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

    h, w = original_shape
    probability = cv2.resize(
        probability,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)

    mask = (probability >= settings.segmentation_threshold).astype(np.uint8)
    return mask, np.clip(probability, 0.0, 1.0)


def _largest_component(binary: np.ndarray) -> np.ndarray:
    binary = (binary > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(component_areas))
    return (labels == largest_label).astype(np.uint8)


def predict_demo(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministic image-processing pipeline for UI demonstration only.
    This is not a trained lesion segmentation model.
    """
    gray = grayscale(image_bgr)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    normalized = enhanced.astype(np.float32) / 255.0

    # Many ultrasound lesions are relatively hypoechoic. This produces a
    # dark-region score and softly prioritizes the central field of view.
    darkness = 1.0 - normalized

    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    sx, sy = max(w * 0.42, 1.0), max(h * 0.42, 1.0)
    central_prior = np.exp(
        -0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2)
    ).astype(np.float32)

    probability = 0.78 * darkness + 0.22 * central_prior
    probability = cv2.GaussianBlur(probability, (0, 0), sigmaX=3.0)
    probability = cv2.normalize(
        probability, None, 0.0, 1.0, cv2.NORM_MINMAX
    ).astype(np.float32)

    threshold = max(float(np.percentile(probability, 84)), 0.55)
    mask = (probability >= threshold).astype(np.uint8)

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = _largest_component(mask)

    return mask, probability


def segment(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    settings = get_settings()

    if settings.model_path:
        checkpoint = Path(settings.model_path)
        if checkpoint.is_file():
            mask, probability = predict_with_checkpoint(image_bgr, checkpoint)
            return mask, probability, "checkpoint"

    if not settings.allow_demo_mode:
        raise RuntimeError(
            "No valid model checkpoint is configured and demo mode is disabled."
        )

    mask, probability = predict_demo(image_bgr)
    return mask, probability, "demo-heuristic"

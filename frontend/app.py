from __future__ import annotations

import io
import os

import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image


API_URL = os.getenv("SONOLESION_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="SonoLesion-AI",
    page_icon="🩻",
    layout="wide",
)

st.title("🩻 SonoLesion-AI")
st.caption(
    "Breast ultrasound lesion segmentation, visualization, and quantitative analysis"
)

st.warning(
    "Research-use prototype only. Not for diagnosis, screening, treatment decisions, or clinical care."
)

with st.sidebar:
    st.header("System status")
    st.code(API_URL)

    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        health_response.raise_for_status()
        health = health_response.json()

        st.success("Backend connected")
        if health.get("model_checkpoint_configured"):
            st.info("Checkpoint mode available")
        elif health.get("demo_mode_allowed"):
            st.info("Demo heuristic mode")
    except Exception:
        st.error("Backend unavailable. Start FastAPI first.")

    st.header("Supported images")
    st.markdown("PNG, JPG, JPEG, BMP, TIFF")


st.subheader("1. Upload ultrasound image")

uploaded = st.file_uploader(
    "Breast ultrasound image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
)

spacing_enabled = st.checkbox(
    "I know the image pixel spacing",
    help="Enable only if you know the physical spacing in mm/pixel.",
)
pixel_spacing = None
if spacing_enabled:
    pixel_spacing = st.number_input(
        "Pixel spacing (mm/pixel)",
        min_value=0.001,
        max_value=10.0,
        value=0.10,
        step=0.01,
        format="%.3f",
    )

if uploaded is not None:
    st.image(uploaded, caption="Uploaded ultrasound image", width="stretch")

if st.button(
    "Analyze lesion",
    type="primary",
    disabled=uploaded is None,
):
    files = {
        "image": (
            uploaded.name,
            uploaded.getvalue(),
            uploaded.type or "application/octet-stream",
        )
    }
    data = {}
    if pixel_spacing is not None:
        data["pixel_spacing_mm"] = str(pixel_spacing)

    with st.spinner("Running lesion analysis..."):
        try:
            response = requests.post(
                f"{API_URL}/api/v1/analyze",
                files=files,
                data=data,
                timeout=300,
            )
            response.raise_for_status()
            st.session_state["sonolesion_result"] = response.json()
        except requests.RequestException as exc:
            detail = ""
            if getattr(exc, "response", None) is not None:
                try:
                    detail = exc.response.json().get("detail", "")
                except Exception:
                    detail = exc.response.text
            st.error(f"Analysis failed. {detail or exc}")


result = st.session_state.get("sonolesion_result")

if result:
    case_id = result["case_id"]
    mode = result["inference_mode"]
    metrics = result["metrics"]

    st.divider()
    st.subheader("2. Analysis result")

    if mode == "demo-heuristic":
        st.warning(
            "This run used the built-in deterministic demonstration heuristic. "
            "The displayed region is for software demonstration only and is not "
            "a trained-model prediction."
        )
    else:
        st.success("A configured PyTorch checkpoint was used.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Lesion area", f"{metrics['area_px2']:.1f} px²")
    m2.metric("Perimeter", f"{metrics['perimeter_px']:.1f} px")
    m3.metric(
        "Equivalent diameter",
        f"{metrics['equivalent_diameter_px']:.1f} px",
    )
    m4.metric("Circularity", f"{metrics['circularity']:.3f}")

    if metrics.get("area_mm2") is not None:
        p1, p2, p3 = st.columns(3)
        p1.metric("Estimated area", f"{metrics['area_mm2']:.2f} mm²")
        p2.metric(
            "Estimated diameter",
            f"{metrics['equivalent_diameter_mm']:.2f} mm",
        )
        p3.metric(
            "Bounding box",
            f"{metrics['bbox_width_mm']:.2f} × {metrics['bbox_height_mm']:.2f} mm",
        )

    st.caption(
        f"Case ID: `{case_id}` | Mode: `{mode}` | "
        f"Lesion fraction: {100 * result['lesion_pixel_fraction']:.2f}% | "
        f"Mean lesion probability: {100 * result['mean_lesion_probability']:.1f}%"
    )

    st.subheader("3. Visual analysis")
    tabs = st.tabs(
        ["Original", "Segmentation mask", "Contour overlay", "Confidence heatmap"]
    )

    kinds = ["original", "mask", "overlay", "heatmap"]
    for tab, kind in zip(tabs, kinds):
        with tab:
            try:
                r = requests.get(
                    f"{API_URL}/api/v1/cases/{case_id}/image/{kind}",
                    timeout=30,
                )
                r.raise_for_status()
                st.image(
                    io.BytesIO(r.content),
                    caption=kind.replace("_", " ").title(),
                    width="stretch",
                )
            except requests.RequestException as exc:
                st.error(f"Could not load {kind}: {exc}")

    st.subheader("4. Morphology summary")
    chart_labels = [
        "Area (px²)",
        "Perimeter (px)",
        "Eq. diameter (px)",
        "BBox width (px)",
        "BBox height (px)",
    ]
    chart_values = [
        metrics["area_px2"],
        metrics["perimeter_px"],
        metrics["equivalent_diameter_px"],
        metrics["bbox_width_px"],
        metrics["bbox_height_px"],
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=chart_labels,
                y=chart_values,
                text=[f"{v:.1f}" for v in chart_values],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        height=420,
        yaxis_title="Value",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("5. Download outputs")
    d1, d2, d3 = st.columns(3)

    for column, kind, label in [
        (d1, "mask", "Download mask"),
        (d2, "overlay", "Download overlay"),
        (d3, "heatmap", "Download heatmap"),
    ]:
        with column:
            try:
                r = requests.get(
                    f"{API_URL}/api/v1/cases/{case_id}/download/{kind}",
                    timeout=30,
                )
                r.raise_for_status()
                st.download_button(
                    label,
                    data=r.content,
                    file_name=f"{case_id}_{kind}.png",
                    mime="image/png",
                    key=f"download_{kind}",
                )
            except requests.RequestException as exc:
                st.error(f"Could not prepare {kind}: {exc}")

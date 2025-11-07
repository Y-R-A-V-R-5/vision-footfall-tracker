# ============================================================
# pages/analysis.py
# Enhanced Model Comparison — Horizontal Layout (Final Version)
# ============================================================

import streamlit as st
import tempfile
import os
import time
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from ultralytics import YOLO

# SORT tracker import
try:
    from sort import Sort
except:
    Sort = None


# -----------------------------
# Custom CSS Styling
# -----------------------------
def inject_custom_css():
    st.markdown("""
    <style>

        /* Each model card container */
        .model-box {
            padding: 12px;
            border-radius: 12px;
            border: 1px solid #e5e5e5;
            background: #fafafa;
            text-align: center;
        }

        /* Model Output Images */
        .result-img {
            width: 35% !important;      /* ✅ keeps results small */
            display: block;
            margin-left: auto;
            margin-right: auto;
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True)


# -----------------------------
# Helper Functions
# -----------------------------
@st.cache_resource
def load_yolo_model(model_path: str):
    """Load YOLOv8 model once (cached)."""
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"⚠️ Could not load model → {model_path} : {e}")
        return None


def analyze_image(image, model, conf_thresh=0.5):
    """Runs inference on an image and returns annotated output + metrics."""
    start = time.time()

    if isinstance(image, Image.Image):
        image = np.array(image)

    results = model.predict(image, conf=conf_thresh, verbose=False)
    inference_time = (time.time() - start) * 1000

    detections = []
    confidences = []

    if results and hasattr(results[0], "boxes"):
        for box in results[0].boxes:
            conf = float(box.conf)
            confidences.append(conf)
            detections.append(conf)

    annotated = results[0].plot()

    return annotated, {
        "detections": len(detections),
        "avg_confidence": np.mean(confidences) if confidences else 0.0,
        "inference_time": inference_time
    }


# -----------------------------
# MAIN PAGE UI
# -----------------------------
def render_page():
    inject_custom_css()

    st.title("🧪 Model Comparison — YOLOv8")

    media_file = st.file_uploader(
        "Upload image",
        type=['jpg', 'jpeg', 'png'],
        help="Image only (video tracking handled on next version)"
    )

    if not media_file:
        st.info("Upload an image to start comparing models…")
        return

    # Display uploaded image
    image = Image.open(media_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.markdown("---")

    # Model selection
    MODEL_PATHS = {
        "YOLOv8n (Nano)": "yolov8n.pt",
        "YOLOv8s (Small)": "yolov8s.pt",
        "YOLOv8m (Medium)": "yolov8m.pt"
    }

    selected_models = st.multiselect(
        "Select models to run",
        list(MODEL_PATHS.keys()),
        help="You can select 1–3 models"
    )

    if not selected_models:
        st.warning("Choose at least one model.")
        return

    st.markdown("---")

    if st.button("🚀 Run Comparison", use_container_width=True):

        comparison_data = []

        # 🔥 Create column layout dynamically based on model count
        cols = st.columns(len(selected_models))

        for col, m in zip(cols, selected_models):

            model = load_yolo_model(MODEL_PATHS[m])
            if model is None:
                continue

            with col:
                st.markdown("<div class='model-box'>", unsafe_allow_html=True)

                st.subheader(m)

                # ✅ Each model has its own confidence input
                conf_val = st.slider(
                    f"Confidence → {m}",
                    0.1, 1.0, 0.5, 0.05,
                    key=f"conf_{m}"
                )

                with st.spinner(f"Analyzing with {m}…"):
                    annotated, metrics = analyze_image(image, model, conf_val)

                # ✅ Image 35% width
                st.image(
                    annotated,
                    caption=f"Detections: {metrics['detections']}",
                    use_container_width=False,
                    output_format="PNG",
                    className="result-img"
                )

                # Metrics
                st.write(f"**Detections:** {metrics['detections']}")
                st.write(f"**Avg Confidence:** `{metrics['avg_confidence']:.3f}`")
                st.write(f"**Inference Time:** `{metrics['inference_time']:.1f} ms`")

                st.markdown("</div>", unsafe_allow_html=True)

                # Add to summary table
                comparison_data.append({
                    "Model": m,
                    "Confidence Used": conf_val,
                    "Detections": metrics["detections"],
                    "Avg Confidence": f"{metrics['avg_confidence']:.3f}",
                    "Inference (ms)": f"{metrics['inference_time']:.1f}"
                })

        st.markdown("---")

        # Comparison Table
        if len(comparison_data) > 1:
            st.subheader("📊 Comparison Summary")
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)


# End of analysis.py
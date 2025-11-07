# ============================================================
# pages/analysis.py
# ------------------------------------------------------------
# Test & Analysis Page for Images and Videos
# - Multi-model comparison
# - Image: Detection only (no tracking)
# - Video: Detection + Tracking + Entry/Exit Counting
# ============================================================

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
import time
from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort  # Install: pip install deep-sort-realtime

def render_page():
    st.markdown("""
    <style>
    .analysis-header {
        background: linear-gradient(90deg, #2B7A78, #3AAFA9);
        color: white;
        padding: 25px 0;
        border-radius: 10px;
        text-align: center;
        font-size: 28px;
        font-weight: 700;
    }
    </style>
    <div class="analysis-header">🧪 Test & Analyze Models</div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================================
    # SECTION 1: Upload Media (Image or Video)
    # ============================================================
    st.markdown("### 📤 Upload Media for Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose an image or video file",
        type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'],
        help="Upload an image for detection analysis or video for tracking + counting"
    )
    
    if uploaded_file is None:
        st.info("👆 Please upload an image or video to begin analysis")
        return
    
    # Determine file type
    file_extension = uploaded_file.name.split('.')[-1].lower()
    is_video = file_extension in ['mp4', 'avi', 'mov']
    
    # Display uploaded media
    st.markdown("---")
    st.markdown("### 📋 Uploaded Media")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if is_video:
            st.video(uploaded_file)
            st.caption(f"📹 Video: {uploaded_file.name}")
        else:
            st.image(uploaded_file, use_container_width =True)
            st.caption(f"🖼️ Image: {uploaded_file.name}")
    
    with col2:
        st.info(f"""
        **File Information:**
        - Type: {'Video' if is_video else 'Image'}
        - Format: {file_extension.upper()}
        - Size: {uploaded_file.size / 1024:.2f} KB
        """)
    
    st.markdown("---")
    
    # ============================================================
    # SECTION 2: Model Selection & Configuration
    # ============================================================
    st.markdown("### 🎛️ Model Selection & Configuration")
    
    # TODO: Replace with actual model paths from your folders
    # Example structure:
    # SOTA_MODELS = {
    #     "YOLOv8n (Nano)": "sota/yolov8n.pt",
    #     "YOLOv8s (Small)": "sota/yolov8s.pt",
    #     "YOLOv8m (Medium)": "sota/yolov8m.pt",
    # }
    # CUSTOM_MODELS = {
    #     "Custom YOLOv8n - JHU": "models/yolov8n_jhu.pt",
    #     "Custom YOLOv8s - CCTV": "models/yolov8s_cctv.pt",
    # }
    
    # Placeholder for model selection
    st.warning("⚠️ **Model Configuration Required**: Please update model paths in the code")
    
    model_options = {
        "YOLOv8n (SOTA)": r"C:\Users\Kapil IT Skill HUB\Desktop\ffc\models\V8m.pt",  # REPLACE WITH ACTUAL PATH
        # "YOLOv8s (SOTA)": "path/to/sota/yolov8s.pt",
        # "Custom YOLOv8n": "path/to/models/custom_yolov8n.pt",
    }
    
    if not model_options:
        st.error("❌ No models available. Please add model paths in the code.")
        st.code("""
# Add your model paths like this:
model_options = {
    "YOLOv8n (SOTA)": "sota/yolov8n.pt",
    "YOLOv8s (SOTA)": "sota/yolov8s.pt",
    "Custom YOLOv8n - JHU": "models/yolov8n_jhu_trained.pt",
    "Custom YOLOv8s - CCTV": "models/yolov8s_cctv_trained.pt",
}
        """, language="python")
        return
    
    selected_models = st.multiselect(
        "Select models to compare",
        options=list(model_options.keys()),
        help="Choose one or more models for comparison analysis"
    )
    
    if not selected_models:
        st.warning("⚠️ Please select at least one model")
        return
    
    # Confidence sliders for each model
    st.markdown("#### 🎚️ Confidence Thresholds")
    confidence_thresholds = {}
    
    cols = st.columns(len(selected_models))
    for idx, model_name in enumerate(selected_models):
        with cols[idx]:
            confidence_thresholds[model_name] = st.slider(
                f"{model_name}",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key=f"conf_{model_name}"
            )
    
    # ROI Configuration for Video
    if is_video:
        st.markdown("#### 🎯 ROI Configuration (for counting)")
        roi_position = st.selectbox(
            "ROI Line Position",
            options=["Center Horizontal", "Center Vertical", "Top", "Bottom", "Left", "Right"],
            help="Position of virtual counting line"
        )
    
    st.markdown("---")
    
    # ============================================================
    # SECTION 3: Run Analysis Button
    # ============================================================
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")
        
        # Progress indicator
        with st.spinner("🔄 Processing... Please wait"):
            
            if is_video:
                # ============================================================
                # VIDEO PROCESSING: Detection + Tracking + Counting
                # ============================================================
                process_video_analysis(
                    tmp_path, 
                    selected_models, 
                    model_options, 
                    confidence_thresholds,
                    roi_position
                )
            else:
                # ============================================================
                # IMAGE PROCESSING: Detection Only
                # ============================================================
                process_image_analysis(
                    tmp_path, 
                    selected_models, 
                    model_options, 
                    confidence_thresholds
                )


# ============================================================
# IMAGE ANALYSIS FUNCTION
# ============================================================
def process_image_analysis(image_path, selected_models, model_options, confidence_thresholds):
    """
    Process image with multiple models for comparison
    No tracking, no entry/exit - only detection
    """
    
    # Read image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create columns for each model
    cols = st.columns(len(selected_models))
    
    for idx, model_name in enumerate(selected_models):
        with cols[idx]:
            st.markdown(f"#### {model_name}")
            
            try:
                # Load model
                model_path = model_options[model_name]
                model = YOLO(model_path)
                
                # Run inference
                start_time = time.time()
                results = model(img, conf=confidence_thresholds[model_name], verbose=False)[0]
                inference_time = time.time() - start_time
                
                # Get detections
                boxes = results.boxes
                num_detections = len(boxes)
                
                # Draw bounding boxes
                annotated_img = results.plot()
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                
                # Display processed image
                st.image(annotated_img_rgb, use_container_width =True)
                
                # Calculate average confidence
                if num_detections > 0:
                    avg_confidence = float(boxes.conf.mean())
                else:
                    avg_confidence = 0.0
                
                # Display metrics
                st.markdown("""
                <style>
                .metric-box {
                    background-color: #DEF2F1;
                    border-radius: 8px;
                    padding: 12px;
                    margin: 8px 0;
                }
                </style>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-box">
                    <b>📊 Performance Metrics:</b><br>
                    • Persons Detected: <b>{num_detections}</b><br>
                    • Avg Confidence: <b>{avg_confidence:.2%}</b><br>
                    • Inference Time: <b>{inference_time*1000:.1f}ms</b>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error processing with {model_name}: {str(e)}")
                st.info("💡 Make sure the model path is correct and the model file exists")


# ============================================================
# VIDEO ANALYSIS FUNCTION
# ============================================================
def process_video_analysis(video_path, selected_models, model_options, confidence_thresholds, roi_position):
    """
    Process video with multiple models for comparison
    Includes: Detection + Tracking + Entry/Exit Counting
    """
    
    # Create columns for each model
    cols = st.columns(len(selected_models))
    
    for idx, model_name in enumerate(selected_models):
        with cols[idx]:
            st.markdown(f"#### {model_name}")
            
            try:
                # Load model
                model_path = model_options[model_name]
                model = YOLO(model_path)
                
                # Initialize tracker (DeepSORT)
                # tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
                # TODO: Implement DeepSORT tracking
                
                # Open video
                cap = cv2.VideoCapture(video_path)
                
                # Get video properties
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Define ROI line based on position
                roi_line = get_roi_line(width, height, roi_position)
                
                # Prepare output video
                output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                # Tracking variables
                entry_count = 0
                exit_count = 0
                tracked_ids = set()
                previous_positions = {}
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                frame_idx = 0
                total_confidence = 0
                total_detections = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Run detection
                    results = model(frame, conf=confidence_thresholds[model_name], verbose=False)[0]
                    boxes = results.boxes
                    
                    # TODO: Implement tracking and counting logic
                    # For now, just annotate detections
                    annotated_frame = results.plot()
                    
                    # Draw ROI line
                    cv2.line(annotated_frame, roi_line[0], roi_line[1], (0, 255, 0), 3)
                    
                    # Calculate statistics
                    if len(boxes) > 0:
                        total_confidence += float(boxes.conf.mean())
                        total_detections += 1
                    
                    # Write frame
                    out.write(annotated_frame)
                    
                    # Update progress
                    frame_idx += 1
                    progress = int((frame_idx / total_frames) * 100)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_idx}/{total_frames}")
                
                cap.release()
                out.release()
                
                progress_bar.empty()
                status_text.empty()
                
                # Display processed video
                st.video(output_path)
                
                # Calculate metrics
                avg_confidence = total_confidence / total_detections if total_detections > 0 else 0.0
                
                # Display metrics
                st.markdown(f"""
                <div class="metric-box">
                    <b>📊 Performance Metrics:</b><br>
                    • Total Frames: <b>{total_frames}</b><br>
                    • Entries: <b>{entry_count}</b><br>
                    • Exits: <b>{exit_count}</b><br>
                    • Unique Persons: <b>{len(tracked_ids)}</b><br>
                    • Avg Confidence: <b>{avg_confidence:.2%}</b><br>
                    • FPS: <b>{fps}</b>
                </div>
                """, unsafe_allow_html=True)
                
                st.success("✅ Video processing complete!")
                
            except Exception as e:
                st.error(f"❌ Error processing with {model_name}: {str(e)}")
                st.info("💡 Make sure the model path is correct and dependencies are installed")


# ============================================================
# HELPER FUNCTION: Get ROI Line Coordinates
# ============================================================
def get_roi_line(width, height, position):
    """
    Returns ROI line coordinates based on position
    Returns: ((x1, y1), (x2, y2))
    """
    if position == "Center Horizontal":
        return ((0, height // 2), (width, height // 2))
    elif position == "Center Vertical":
        return ((width // 2, 0), (width // 2, height))
    elif position == "Top":
        return ((0, height // 4), (width, height // 4))
    elif position == "Bottom":
        return ((0, 3 * height // 4), (width, 3 * height // 4))
    elif position == "Left":
        return ((width // 4, 0), (width // 4, height))
    elif position == "Right":
        return ((3 * width // 4, 0), (3 * width // 4, height))
    else:
        return ((0, height // 2), (width, height // 2))
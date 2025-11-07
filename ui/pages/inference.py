# ============================================================
# pages/inference.py
# Real-time Webcam Inference with Performance Monitoring
# ============================================================

import streamlit as st
import cv2
import numpy as np
import time
from collections import deque
from ultralytics import YOLO
import tempfile
import os

# SORT tracker
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
    /* Live indicator */
    .live-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: #ff0000;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    
    /* Performance card */
    .perf-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
    }
    
    /* Status indicators */
    .status-active {
        background-color: #28a745;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
    }
    
    .status-inactive {
        background-color: #6c757d;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
    }
    
    /* Control buttons */
    .control-btn {
        margin: 5px;
    }
    
    /* Metric display */
    .metric-display {
        background-color: #f8f9fa;
        border-left: 4px solid #3AAFA9;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# Session State Initialization
# -----------------------------
def initialize_session_state():
    """Initialize all session state variables."""
    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "entries" not in st.session_state:
        st.session_state.entries = 0
    if "exits" not in st.session_state:
        st.session_state.exits = 0
    if "tracked_ids" not in st.session_state:
        st.session_state.tracked_ids = {}
    if "fps_history" not in st.session_state:
        st.session_state.fps_history = deque(maxlen=30)
    if "conf_history" not in st.session_state:
        st.session_state.conf_history = deque(maxlen=30)

# -----------------------------
# Model Loading
# -----------------------------
@st.cache_resource
def load_model(model_path):
    """Load YOLO model with caching."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

# -----------------------------
# Webcam Processing Function
# -----------------------------
def process_webcam_stream(
    model,
    device_id=0,
    conf_thresh=0.5,
    roi_orientation="Horizontal",
    enable_tracking=True,
    tracker_type="SORT"
):
    """Process webcam stream with real-time inference."""
    
    # Open webcam
    cap = cv2.VideoCapture(device_id)
    
    if not cap.isOpened():
        st.error("❌ Could not open webcam. Please check your camera connection.")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get frame dimensions
    ret, frame = cap.read()
    if not ret:
        st.error("❌ Could not read from webcam")
        cap.release()
        return
    
    h, w = frame.shape[:2]
    roi_line = int(h / 2) if roi_orientation == "Horizontal" else int(w / 2)
    
    # Initialize tracker
    tracker = None
    if enable_tracking and tracker_type == "SORT" and Sort:
        tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    
    # Create placeholders
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 📹 Live Feed")
        video_placeholder = st.empty()
    
    with col2:
        st.markdown("### 📊 Real-time Metrics")
        fps_metric = st.empty()
        conf_metric = st.empty()
        det_metric = st.empty()
        entry_metric = st.empty()
        exit_metric = st.empty()
        
        st.markdown("---")
        st.markdown("### 📈 Performance")
        fps_chart = st.empty()
        conf_chart = st.empty()
    
    # Control buttons
    col_stop, col_reset = st.columns(2)
    with col_stop:
        stop_button = st.button("⏹️ Stop Camera", use_container_width=True, type="primary")
    with col_reset:
        reset_button = st.button("🔄 Reset Counts", use_container_width=True)
    
    # Reset counts if button clicked
    if reset_button:
        st.session_state.entries = 0
        st.session_state.exits = 0
        st.session_state.tracked_ids = {}
        st.success("✅ Counts reset!")
    
    # FPS calculation variables
    fps_history = deque(maxlen=30)
    conf_history = deque(maxlen=30)
    frame_count = 0
    
    # Main processing loop
    while st.session_state.camera_active and not stop_button:
        frame_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Failed to grab frame")
            break
        
        # Run inference
        results = model.predict(frame, conf=conf_thresh, verbose=False)
        
        # Extract detections
        dets = []
        confidences = []
        
        if results and hasattr(results[0], "boxes") and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf)
                dets.append([x1, y1, x2, y2, conf])
                confidences.append(conf)
        
        dets = np.array(dets)
        
        # Tracking
        if tracker and enable_tracking and len(dets) > 0:
            tracks = tracker.update(dets)
        else:
            tracks = dets
        
        # Process tracks and count
        for t in tracks:
            if len(t) >= 5:
                x1, y1, x2, y2, track_id = map(int, t[:5])
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if enable_tracking:
                    cv2.putText(frame, f'ID:{track_id}', (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Calculate center
                center_y = int((y1 + y2) / 2)
                center_x = int((x1 + x2) / 2)
                
                # Tracking and counting logic
                if enable_tracking and track_id not in st.session_state.tracked_ids:
                    st.session_state.tracked_ids[track_id] = {
                        "prev_pos": (center_x, center_y),
                        "counted": False
                    }
                
                if enable_tracking and not st.session_state.tracked_ids[track_id]["counted"]:
                    prev_pos = st.session_state.tracked_ids[track_id]["prev_pos"]
                    
                    if roi_orientation == "Horizontal":
                        if prev_pos[1] < roi_line and center_y > roi_line:
                            st.session_state.exits += 1
                            st.session_state.tracked_ids[track_id]["counted"] = True
                        elif prev_pos[1] > roi_line and center_y < roi_line:
                            st.session_state.entries += 1
                            st.session_state.tracked_ids[track_id]["counted"] = True
                    else:
                        if prev_pos[0] < roi_line and center_x > roi_line:
                            st.session_state.exits += 1
                            st.session_state.tracked_ids[track_id]["counted"] = True
                        elif prev_pos[0] > roi_line and center_x < roi_line:
                            st.session_state.entries += 1
                            st.session_state.tracked_ids[track_id]["counted"] = True
                    
                    st.session_state.tracked_ids[track_id]["prev_pos"] = (center_x, center_y)
        
        # Draw ROI line
        if roi_orientation == "Horizontal":
            cv2.line(frame, (0, roi_line), (w, roi_line), (0, 0, 255), 2)
        else:
            cv2.line(frame, (roi_line, 0), (roi_line, h), (0, 0, 255), 2)
        
        # Calculate FPS
        frame_time = time.time() - frame_start
        fps = 1 / frame_time if frame_time > 0 else 0
        fps_history.append(fps)
        
        # Calculate average confidence
        avg_conf = np.mean(confidences) if confidences else 0.0
        conf_history.append(avg_conf)
        
        # Draw FPS on frame
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Detections: {len(dets)}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if enable_tracking:
            cv2.putText(frame, f'Entries: {st.session_state.entries} | Exits: {st.session_state.exits}',
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Update display
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # Update metrics
        avg_fps = np.mean(list(fps_history)) if fps_history else 0
        fps_metric.metric("🎯 FPS", f"{avg_fps:.1f}", delta=f"{fps:.1f}")
        conf_metric.metric("🎯 Avg Confidence", f"{avg_conf:.3f}")
        det_metric.metric("🔍 Detections", len(dets))
        
        if enable_tracking:
            entry_metric.metric("↗️ Entries", st.session_state.entries, delta="↑")
            exit_metric.metric("↙️ Exits", st.session_state.exits, delta="↓")
        
        # Update charts
        if len(fps_history) > 1:
            fps_chart.line_chart(list(fps_history), height=150)
        if len(conf_history) > 1:
            conf_chart.line_chart(list(conf_history), height=150)
        
        frame_count += 1
        
        # Small delay to prevent overwhelming
        time.sleep(0.01)
    
    # Cleanup
    cap.release()
    st.session_state.camera_active = False
    st.info("📹 Camera stopped")

# -----------------------------
# Main Render Function
# -----------------------------
def render_page():
    inject_custom_css()
    initialize_session_state()
    
    # Page Header
    st.markdown("""
    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 10px; margin-bottom: 20px;'>
        <h1 style='color: white; text-align: center; margin: 0;'>📹 Live Inference</h1>
        <p style='color: white; text-align: center; margin-top: 10px; font-size: 16px;'>
            Real-time detection and counting using your webcam
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Camera status indicator
    if st.session_state.camera_active:
        st.markdown("""
        <div style='text-align: center; padding: 10px; background-color: #d4edda; border-radius: 8px; margin-bottom: 20px;'>
            <span class='live-indicator'></span>
            <span style='margin-left: 10px; font-weight: 600; color: #155724;'>LIVE - Camera Active</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align: center; padding: 10px; background-color: #f8d7da; border-radius: 8px; margin-bottom: 20px;'>
            <span style='font-weight: 600; color: #721c24;'>⏹️ Camera Inactive</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Configuration Section
    st.markdown("### ⚙️ Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        MODEL_OPTIONS = {
            "YOLOv8n (Fastest)": "yolov8n.pt",
            "YOLOv8s (Balanced)": "yolov8s.pt",
            "YOLOv8m (Accurate)": "yolov8m.pt"
        }
        selected_model = st.selectbox(
            "Model",
            list(MODEL_OPTIONS.keys()),
            help="Select YOLOv8 model variant"
        )
        model_path = MODEL_OPTIONS[selected_model]
    
    with col2:
        device_id = st.number_input(
            "Camera Device ID",
            min_value=0,
            max_value=10,
            value=0,
            help="Usually 0 for built-in webcam, 1 for external"
        )
    
    with col3:
        conf_thresh = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        enable_tracking = st.checkbox("Enable Tracking", value=True)
    
    with col5:
        tracker_type = st.selectbox(
            "Tracker",
            ["SORT", "DeepSORT"],
            disabled=not enable_tracking
        )
    
    with col6:
        roi_orientation = st.radio(
            "ROI Orientation",
            ["Horizontal", "Vertical"],
            disabled=not enable_tracking
        )
    
    st.markdown("---")
    
    # Control Buttons
    col_start, col_info = st.columns([1, 3])
    
    with col_start:
        if not st.session_state.camera_active:
            if st.button("🎥 Start Camera", type="primary", use_container_width=True):
                # Load model
                with st.spinner("Loading model..."):
                    model = load_model(model_path)
                
                if model:
                    st.session_state.camera_active = True
                    st.session_state.model_loaded = True
                    st.rerun()
    
    with col_info:
        if not st.session_state.camera_active:
            st.info("👆 Click 'Start Camera' to begin real-time inference")
        else:
            st.warning("⚡ Camera is running. Processing frames in real-time...")
    
    # Process camera stream if active
    if st.session_state.camera_active and st.session_state.model_loaded:
        model = load_model(model_path)
        if model:
            process_webcam_stream(
                model=model,
                device_id=device_id,
                conf_thresh=conf_thresh,
                roi_orientation=roi_orientation,
                enable_tracking=enable_tracking,
                tracker_type=tracker_type
            )
    
    # Instructions
    if not st.session_state.camera_active:
        st.markdown("---")
        st.markdown("### 📖 Instructions")
        st.markdown("""
        1. **Select Model**: Choose a YOLOv8 variant based on your needs
           - **Nano (n)**: Fastest, best for real-time on CPU
           - **Small (s)**: Balanced speed and accuracy
           - **Medium (m)**: Most accurate, requires better hardware
        
        2. **Configure Camera**: Set device ID (usually 0 for built-in webcam)
        
        3. **Adjust Settings**: Set confidence threshold and tracking preferences
        
        4. **Start Camera**: Click the button to begin real-time inference
        
        5. **Monitor Performance**: Watch FPS, detections, and counting metrics
        
        6. **Reset Counts**: Use the reset button to clear entry/exit counts
        
        7. **Stop Camera**: Click stop when finished
        
        ---
        
        **Tips for Best Performance:**
        - Use YOLOv8n for real-time CPU inference
        - Lower confidence threshold for more detections
        - Ensure good lighting for better detection
        - Position camera to have clear view of heads
        - Horizontal ROI works best for doorways
        - Vertical ROI works best for corridors
        """)

# End of inference.py
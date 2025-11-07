# ============================================================
# pages/inference.py
# ------------------------------------------------------------
# Live Inference Page
# - Real-time camera feed processing
# - DeepSORT tracking
# - Entry/Exit counting with ROI
# - Performance metrics display
# ============================================================

import streamlit as st
import cv2
import numpy as np
from collections import defaultdict, deque
import time
from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort  # Install: pip install deep-sort-realtime

def render_page():
    st.markdown("""
    <style>
    .inference-header {
        background: linear-gradient(90deg, #2B7A78, #3AAFA9);
        color: white;
        padding: 25px 0;
        border-radius: 10px;
        text-align: center;
        font-size: 28px;
        font-weight: 700;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .metric-value {
        font-size: 36px;
        font-weight: 700;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 14px;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
    <div class="inference-header">📹 Live Inference</div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================================
    # SECTION 1: Camera Detection & Status
    # ============================================================
    st.markdown("### 📷 Camera Status")
    
    # Check camera availability
    camera_available = check_camera_availability()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if camera_available:
            st.success("✅ Camera detected and ready")
        else:
            st.error("❌ No camera detected")
            st.warning("""
            **Troubleshooting:**
            - Ensure your camera is connected
            - Check camera permissions in browser/system settings
            - Try using a different camera source
            - For USB cameras, ensure drivers are installed
            """)
            return
    
    with col2:
        st.info(f"""
        **Camera Info:**
        - Source: Default Camera
        - Status: {'Active' if camera_available else 'Inactive'}
        """)
    
    st.markdown("---")
    
    # ============================================================
    # SECTION 2: Model Selection & Configuration
    # ============================================================
    st.markdown("### ⚙️ Configuration")
    
    col_conf1, col_conf2 = st.columns([1, 1])
    
    with col_conf1:
        st.markdown("#### 🤖 Model Selection")
        
        # TODO: Replace with actual model paths
        model_options = {
        "YOLOv8n (SOTA)": r"C:\Users\Kapil IT Skill HUB\Desktop\ffc\models\V8m.pt",  # REPLACE WITH ACTUAL PATH
            # "YOLOv8s (SOTA)": "sota/yolov8s.pt",
            # "Custom YOLOv8n": "models/custom_yolov8n.pt",
        }
        
        if not model_options:
            st.error("❌ No models available. Please add model paths in the code.")
            st.code("""
# Add your model paths like this:
model_options = {
    "YOLOv8n (SOTA)": "sota/yolov8n.pt",
    "Custom YOLOv8n": "models/yolov8n_trained.pt",
}
            """, language="python")
            return
        
        selected_model = st.selectbox(
            "Choose detection model",
            options=list(model_options.keys()),
            help="Select a model for real-time inference"
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
    
    with col_conf2:
        st.markdown("#### 🎯 ROI Configuration")
        
        roi_position = st.selectbox(
            "Counting Line Position",
            options=["Center Horizontal", "Center Vertical", "Top", "Bottom", "Left", "Right"],
            help="Position of virtual counting line for entry/exit detection"
        )
        
        st.markdown("#### 🎬 Display Options")
        show_tracking_ids = st.checkbox("Show Tracking IDs", value=True)
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
    
    st.markdown("---")
    
    # ============================================================
    # SECTION 3: Live Feed & Controls
    # ============================================================
    st.markdown("### 🎥 Live Feed")
    
    # Initialize session state
    if 'inference_running' not in st.session_state:
        st.session_state.inference_running = False
    if 'entry_count' not in st.session_state:
        st.session_state.entry_count = 0
    if 'exit_count' not in st.session_state:
        st.session_state.exit_count = 0
    
    # Control buttons
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("▶️ Start Inference", type="primary", use_container_width=True):
            st.session_state.inference_running = True
            st.rerun()
    
    with col_btn2:
        if st.button("⏸️ Stop Inference", use_container_width=True):
            st.session_state.inference_running = False
            st.rerun()
    
    with col_btn3:
        if st.button("🔄 Reset Counters", use_container_width=True):
            st.session_state.entry_count = 0
            st.session_state.exit_count = 0
            st.rerun()
    
    st.markdown("---")
    
    # ============================================================
    # SECTION 4: Live Metrics Dashboard
    # ============================================================
    st.markdown("### 📊 Real-Time Metrics")
    
    # Metrics display (will be updated during inference)
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <div class="metric-label">Entries</div>
            <div class="metric-value">{st.session_state.get('entry_count', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="metric-label">Exits</div>
            <div class="metric-value">{st.session_state.get('exit_count', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div class="metric-label">Current</div>
            <div class="metric-value">{st.session_state.get('current_count', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <div class="metric-label">FPS</div>
            <div class="metric-value">{st.session_state.get('fps', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============================================================
    # SECTION 5: Run Inference Loop
    # ============================================================
    if st.session_state.inference_running:
        
        # Video placeholder
        video_placeholder = st.empty()
        
        # Additional metrics
        metrics_placeholder = st.empty()
        
        try:
            # Load model
            model_path = model_options[selected_model]
            model = YOLO(model_path)
            
            # Initialize tracker
            # tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
            # TODO: Implement DeepSORT
            
            # Open camera
            cap = cv2.VideoCapture(0)  # 0 for default camera
            
            if not cap.isOpened():
                st.error("❌ Failed to open camera")
                st.session_state.inference_running = False
                return
            
            # Get frame properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Define ROI line
            roi_line = get_roi_line(width, height, roi_position)
            
            # Tracking variables
            tracked_objects = {}
            previous_positions = {}
            unique_ids = set()
            confidence_history = deque(maxlen=30)
            fps_history = deque(maxlen=30)
            
            frame_count = 0
            start_time = time.time()
            
            while st.session_state.inference_running:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to read frame from camera")
                    break
                
                frame_start = time.time()
                
                # Run detection
                results = model(frame, conf=confidence_threshold, verbose=False)[0]
                boxes = results.boxes
                
                # TODO: Implement tracking logic with DeepSORT
                # For now, just draw detections
                annotated_frame = results.plot()
                
                # Draw ROI line
                cv2.line(annotated_frame, roi_line[0], roi_line[1], (0, 255, 0), 3)
                cv2.putText(annotated_frame, "ROI Line", 
                           (roi_line[0][0] + 10, roi_line[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Calculate FPS
                frame_time = time.time() - frame_start
                fps = 1 / frame_time if frame_time > 0 else 0
                fps_history.append(fps)
                avg_fps = sum(fps_history) / len(fps_history)
                
                # Calculate average confidence
                if len(boxes) > 0:
                    avg_conf = float(boxes.conf.mean())
                    confidence_history.append(avg_conf)
                
                # Display FPS on frame
                cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display counts on frame
                cv2.putText(annotated_frame, f"Entries: {st.session_state.entry_count}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Exits: {st.session_state.exit_count}", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Convert to RGB for display
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                video_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
                
                # Update metrics
                st.session_state.fps = int(avg_fps)
                st.session_state.current_count = len(boxes)
                
                # Display additional metrics
                with metrics_placeholder.container():
                    metric_info_cols = st.columns(3)
                    
                    with metric_info_cols[0]:
                        avg_confidence = sum(confidence_history) / len(confidence_history) if confidence_history else 0
                        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                    
                    with metric_info_cols[1]:
                        st.metric("Unique Persons", len(unique_ids))
                    
                    with metric_info_cols[2]:
                        elapsed_time = time.time() - start_time
                        st.metric("Runtime", f"{elapsed_time:.0f}s")
                
                frame_count += 1
                
                # Small delay to prevent overwhelming the UI
                time.sleep(0.01)
            
            cap.release()
            st.success("✅ Inference stopped successfully")
            
        except Exception as e:
            st.error(f"❌ Error during inference: {str(e)}")
            st.info("💡 Make sure the model path is correct and camera is accessible")
            st.session_state.inference_running = False
    
    else:
        st.info("👆 Click 'Start Inference' to begin real-time detection and counting")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_camera_availability():
    """
    Check if camera is available
    """
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.release()
            return True
        return False
    except:
        return False


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


def calculate_line_crossing(prev_y, curr_y, line_y, threshold=5):
    """
    Determine if object crossed the ROI line
    Returns: 'entry', 'exit', or None
    """
    if prev_y < line_y - threshold and curr_y >= line_y:
        return 'entry'
    elif prev_y > line_y + threshold and curr_y <= line_y:
        return 'exit'
    return None


# TODO: Implement DeepSORT tracking integration
# def update_tracking(detections, tracker):
#     """
#     Update tracker with new detections
#     """
#     # Convert YOLO detections to DeepSORT format
#     # Track objects across frames
#     # Return tracked objects with IDs
#     pass
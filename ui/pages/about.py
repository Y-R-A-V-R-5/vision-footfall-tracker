# ============================================================
# pages/about.py
# ------------------------------------------------------------
# Enhanced Overview Page for AI Footfall Counter
# with Detailed Dataset, Training, and Inference Information
# ============================================================

import streamlit as st

def render_page():
    # --- Page Title with Gradient Banner ---
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #2B7A78, #3AAFA9);
        color: white;
        padding: 30px 0;
        border-radius: 10px;
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .sub-header {
        text-align: center;
        font-size: 18px;
        margin-top: 5px;
        color: #555;
    }
    </style>
    <div class="main-header">👣 AI Footfall Counter</div>
    <p class="sub-header">Real-time detection, tracking, and counting using Computer Vision</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Objective Section ---
    st.markdown("### 🎯 Objective")
    st.write("""
    The objective of this project is to **design and implement a computer vision–based system**
    that automatically counts the number of people entering and exiting through a designated area
    such as a doorway, corridor, or gate.

    This project demonstrates an **end-to-end understanding** of:
    - AI model integration (YOLOv8 for head detection)
    - Object tracking (SORT / DeepSORT)
    - Event-based counting logic within a real-world CCTV context.
    """)

    st.markdown("---")

    # --- Dataset Description Section ---
    st.markdown("### 📂 Dataset Description")
    st.markdown("""
    Two main datasets were used to train and refine the head detection model:

    - [**JHU-CROWD Dataset (Kaggle)**](https://www.kaggle.com/datasets/hoangxuanviet/jhu-crowd):  
      Provides **head-level annotations** across diverse and dense crowd scenes.  
      Useful for teaching the model to recognize heads even under occlusion and high crowd density.

    - [**Head Detection CCTV Dataset (Kaggle)**](https://www.kaggle.com/datasets/hoangxuanviet/head-detection-cctv):  
      Offers **1,700 CCTV-style images** from doorway and corridor perspectives, 
      making it more aligned with real-world surveillance applications.
    """)

    st.info("""
    ✅ **Domain Alignment Insight:**  
    The project initially started with JHU-CROWD for variety but later shifted to the 
    Head Detection CCTV dataset to better match **fixed surveillance views**.  
    This transition greatly improved detection stability and counting accuracy.
    """)

    st.markdown("---")

    # --- Data Preparation and Domain Adaptation ---
    st.markdown("### 🧩 Data Preparation & Domain Adaptation")
    st.markdown("""
    To bridge the gap between diverse datasets and real CCTV conditions, 
    the following preprocessing and augmentation strategies were implemented:
    """)

    st.markdown("""
    - **Focus on Head Regions:** Training uses only annotated head areas, removing unnecessary background.  
      Improves generalization across lighting, crowd density, and environments.  
    - **Geometric Transformations:** Rotation, scaling, and perspective shifts to simulate camera angles.  
    - **Photometric Adjustments:** Brightness, contrast, and Gaussian noise to mimic lighting variations.  
    - **Transfer Learning:** Fine-tuning a pretrained YOLOv8 model to accelerate convergence and accuracy.  
    """)

    st.success("""
    💡 *By emphasizing human head features and simulating CCTV conditions, 
    the model learns to recognize people independently of background context.*  
    """)

    st.markdown("---")

    # --- Core Components Section (Stylized Cards) ---
    st.markdown("### 🧠 System Components")

    st.markdown("""
    <style>
    .card-container {
        display: flex;
        justify-content: space-between;
        gap: 20px;
        flex-wrap: wrap;
        margin-top: 20px;
    }
    .card {
        flex: 1;
        min-width: 280px;
        background-color: #DEF2F1;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        transition: 0.3s ease-in-out;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.2);
    }
    .card h4 {
        color: #2B7A78;
        margin-bottom: 10px;
    }
    </style>

    <div class="card-container">
        <div class="card">
            <h4>🧠 Detection</h4>
            <p>Detects human heads using YOLOv8 trained on CCTV-style datasets, ensuring robustness in various environments.</p>
        </div>
        <div class="card">
            <h4>🎯 Tracking</h4>
            <p>Assigns unique IDs using SORT or DeepSORT, maintaining continuity across frames for reliable tracking.</p>
        </div>
        <div class="card">
            <h4>📈 Counting</h4>
            <p>Counts entries/exits via virtual ROI lines, analyzing direction of movement dynamically for real-time analytics.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Training & Validation Section ---
    st.markdown("### ⚙️ Training, Validation, and Testing Strategy")
    st.markdown("""
    - **Training & Validation Merge:**  
      Training and validation splits were combined to increase the number of annotated samples.  
    - **Testing Phase:**  
      The test split was reserved for internal evaluation to ensure model generalization.  
    - **Transfer Learning:**  
      Fine-tuning YOLOv8 weights allowed faster convergence and better feature adaptation.  
    """)

    st.info("""
    🎯 *The goal was not benchmark performance on a dataset — 
    but achieving accurate head detection and counting in real-world CCTV feeds.*
    """)

    st.markdown("---")

    # --- Pipeline Section ---
    st.markdown("### 🔄 Inference & Counting Pipeline")

    st.markdown("""
    <style>
    .pipeline {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 25px;
    }
    .step {
        background-color: #3AAFA9;
        color: white;
        border-radius: 50px;
        padding: 10px 18px;
        font-size: 15px;
        font-weight: 500;
        white-space: nowrap;
    }
    .arrow {
        color: #3AAFA9;
        font-size: 22px;
        font-weight: bold;
    }
    </style>

    <div class="pipeline">
        <div class="step">📹 Frame Capture</div>
        <div class="arrow">➡️</div>
        <div class="step">🧠 Head Detection</div>
        <div class="arrow">➡️</div>
        <div class="step">🎯 Tracking</div>
        <div class="arrow">➡️</div>
        <div class="step">🚪 ROI Line Crossing</div>
        <div class="arrow">➡️</div>
        <div class="step">📊 Counting & Display</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("""
    Each tracked head crossing the virtual ROI contributes to:
    - **Entry count** → upward  movement  
    - **Exit count** → downward movement  
    """)

    st.markdown("---")

    # --- Applications Section ---
    st.markdown("### 🌍 Real-World Applications")
    st.markdown("""
    - 🏬 **Retail Analytics:** Monitor entry and exit activity for footfall estimation.  
    - 🏢 **Smart Buildings:** Automate door control, HVAC systems, and occupancy limits.  
    - 🎟️ **Event Monitoring:** Manage crowd density and flow during large gatherings.  
    - 🚉 **Transport Hubs:** Track passenger flow in stations and terminals.  
    """)

    st.markdown("---")

    # --- Footer / Tip Box ---
    st.markdown("""
    <style>
    .info-box {
        background-color: #DEF2F1;
        border-left: 6px solid #3AAFA9;
        border-radius: 6px;
        padding: 15px;
        font-size: 15px;
        color: #333;
        margin-top: 30px;
    }
    </style>
    <div class="info-box">
        💡 <b>Tip:</b> Use the sidebar or bottom navigation to explore <b>Testing</b> and <b>Live Inference</b> pages.  
        Each section demonstrates detection, tracking, and counting in different scenarios.
    </div>
    """, unsafe_allow_html=True)
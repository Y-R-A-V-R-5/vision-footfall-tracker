# 🧠 Vision Footfall Tracker  

> An intelligent, camera-based analytics system built using **YOLOv8** for head detection and modern object tracking algorithms such as **SORT**, **DeepSORT**, **ByteTrack**, and **BOT-SORT**.  
> Enables real-time counting of entries and exits through a designated Region of Interest (ROI) — perfect for **smart retail**, **facility management**, and **crowd analytics**.

---

## 🎯 Overview  

**Vision Footfall Tracker** is a modular and efficient computer vision solution for real-time human flow analytics.  
It transforms ordinary CCTV feeds into actionable insights — offering detection, tracking, visualization, and export functionalities in one unified tool.

---

## 🚀 Key Features  

- 🎯 **Accurate Detection** — Head-based YOLOv8 detection optimized for CCTV and overhead views.  
- 🔍 **Robust Tracking** — Smooth ID tracking using multiple algorithms (**SORT**, **DeepSORT**, **ByteTrack**, **BOT-SORT**).
- 🧩 **Flexible ROI** — Automatic or manual ROI line definition adaptable to various video dimensions.  
- 📈 **Smart Analytics** — Real-time entry/exit counts and trend visualization.  
- 🧠 **Interactive UI** — Multi-page Streamlit dashboard with configurable controls, live metrics, and export options.
- ⚙️ **Customizable Models** — Side-by-side comparison of **YOLOv8-small** vs **YOLOv8-medium** for accuracy–latency trade-offs.
- 📹 **Multi-Source Support** — Process uploaded videos or live webcam feeds seamlessly.  
- 💾 **Exportable Results** — Save processed videos, CSV summaries, and configuration snapshots.  

---

## 📂 Datasets Used  

| Dataset | Description | Link |
|---------|-------------|------|
| **JHU-CROWD Dataset** | High-density scenes with head-level annotations (roads, public gatherings, swimming pools). Great for learning detection under occlusion and crowding scenarios, but introduces domain mismatch for typical CCTV footfall use cases. | https://www.kaggle.com/datasets/hoangxuanviet/jhu-crowd |
| **Head Detection CCTV Dataset** | ~1,700 images collected from overhead/fixed CCTV angles. Ideal for real-world entry/exit monitoring and doorway tracking, improving model context adaptation and stability. | https://www.kaggle.com/datasets/hoangxuanviet/head-detection-cctv |

---

## 🏗️ Applications  

| Sector | Description |
|:--------|:-------------|
| 🏬 **Retail & Malls** | Track customer inflow and outflow across entry zones. |
| 🏢 **Office Buildings** | Monitor occupancy levels and space utilization. |
| 🏫 **Educational Institutions** | Measure hallway or classroom foot traffic. |
| 🚉 **Transportation Hubs** | Analyze passenger flow and queue density. |
| 🏛️ **Public Venues** | Monitor crowd safety during events. |

---

## ⚙️ Tech Stack  

| Component | Technology |
|------------|-------------|
| **Detection** | YOLOv8 (Ultralytics) |
| **Tracking** | SORT / DeepSORT  |
| **UI** | Streamlit + Plotly |
| **Data Handling** | OpenCV, NumPy, Pandas |
| **Experiment Management** | MLflow |

---

## 🌟 Vision Statement  

> To create an accessible, modular, and efficient computer vision system for real-time human flow analytics under real-world CCTV constraints.

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
- 🧠 **Interactive UI** — Multi-page **Streamlit** dashboard with intuitive controls, live metrics, and export options.  
- ⚙️ **Customizable Models** — Compare **YOLOv8-small** vs **YOLOv8-medium** side-by-side performance.  
- 📹 **Multi-Source Support** — Process uploaded videos or live webcam feeds seamlessly.  
- 💾 **Exportable Results** — Save processed videos, CSV summaries, and configuration snapshots.  

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
| **Tracking** | SORT / DeepSORT / ByteTrack / BOT-SORT |
| **UI** | Streamlit + Plotly |
| **Data Handling** | OpenCV, NumPy, Pandas |
| **Experiment Management** | MLflow |
| **Deployment** | Local and Cloud environments supported |

---

## 🌟 Vision Statement  

> To create an accessible, modular, and efficient computer vision solution for real-time human flow analytics — turning ordinary CCTV feeds into actionable insights.

---

## 🧩 Project Structure  

```plaintext
vision-footfall-tracker/
│
├── app.py                     # Streamlit UI entry point
├── models/                    # YOLOv8 weights and configs
├── trackers/                  # SORT, DeepSORT, ByteTrack, BOT-SORT scripts
├── utils/                     # Helper functions and ROI processors
├── datasets/                  # (Ignored by .gitignore)
├── runs/                      # Generated outputs (ignored)
├── mlruns/                    # MLflow experiments (ignored)
├── requirements.txt            # Dependencies list
└── README.md                   # Project documentation

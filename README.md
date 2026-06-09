# Real-Time-AI-Biomechanics-Squat-Form-Analyzer
An end-to-end computer vision and applied machine learning pipeline that extracts multi-joint human skeletal landmarks from raw video feeds, engineers scale-invariant biomechanical features, and performs real-time edge inference to classify athletic form.


# Real-Time AI Biomechanics & Squat Form Analyzer

An end-to-end computer vision and applied machine learning pipeline that extracts multi-joint human skeletal landmarks from raw video feeds, engineers scale-invariant biomechanical features, and performs real-time edge inference to classify athletic form.



## 🚀 System Architecture & Pipeline

The system is engineered as a decoupled, 5-stage applied machine learning pipeline:
1. **Vision Ingestion:** OpenCV captures high-throughput video streams frame-by-frame on local edge hardware.
2. **Feature Extraction:** Google MediaPipe (Pose Landmarker Task) parses raw pixel arrays into 33 distinct 3D coordinates ($X, Y, Z$) representing human joints.
3. **Biomechanical Vector Engineering:** NumPy translates joint coordinates into spatial vectors, applying the dot product and clipping transformations to yield scale-invariant knee and hip angles.
4. **Tabular Serialization:** Pandas structures the multi-dimensional feature rows along with binary state labels ("Good" / "Bad") into a compiled `.csv` dataset matrix.
5. **Edge Optimization & Classification:** Scikit-Learn processes the dataset via a Random Forest Classifier to map non-linear decision boundaries, saving the trained brain as a serialized `.pkl` file for low-latency live inference.

---

## 🛠️ Tech Stack & Engineering Choices

* **Python 3.11+ / macOS Virtual Environments (`venv`)** — Complete isolation of complex machine learning dependencies to bypass global system contamination blocks (PEP 668).
* **OpenCV (`cv2`)** — Low-latency camera matrix streaming and dynamic frame-by-frame visual Heads-Up Display (HUD) overlay rendering.
* **Google MediaPipe Tasks** — Optimized local TensorFlow Lite deployment utilizing the `pose_landmarker_heavy.task` bundle running on local CPU/M1 architectures (Edge AI).
* **NumPy** — Vector math optimization using standard algebraic coordinate calculations:
* **Pandas** — Automated processing and engineering of multi-column dataframes.
* **Scikit-Learn** — Selected **Random Forest Classifier** over heavy Deep Neural Networks (DNNs) to minimize compute overhead, completely eliminating inference latency on local edge hardware while maximizing model explainability.

---

## 📂 Project Structure

```text
ml_project/
├── venv/                       # Isolated Python Environment Sandbox
├── videos/                     # Data Ingestion Directory
│   ├── good/                   # Drop perfect reference videos here
│   └── bad/                    # Drop faulty or incomplete form videos here
├── advanced_data_collector.py  # Manual real-time webcam data recording tool
├── video_pipeline.py           # Automated bulk video ingestion & extraction pipeline
├── train_advanced_model.py     # Training script with complete evaluation metrics
├── app.py                      # Core live edge inference execution loop
├── pose_landmarker.task        # Downloaded Google TFLite binary weights
├── advanced_squat_model.pkl    # Serialized trained Random Forest classifier
└── README.md                   # System documentation

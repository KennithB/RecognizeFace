# Doorbell Face Recognition System (InsightFace)

Real-time face recognition and stranger detection using InsightFace (`buffalo_l`) with NVIDIA CUDA acceleration via `onnxruntime-gpu`.

## Features
- **InsightFace `buffalo_l` Model**: High-accuracy face detection and 512-d ArcFace embeddings.
- **Real-Time Camera Integration**: Supports USB webcams (indices `0`, `1`, etc.) and IP / RTSP security cameras (`rtsp://user:pass@ip:554/stream`).
- **Stranger / Intruder Detection**: Unknown faces flagged with red bounding box and `STRANGER` alert badge.
- **Local Profile Store**: Face profiles stored as normalized `.npy` embedding vectors in `profiles/` directory (no database needed).
- **Interactive UI**:
  - Live surveillance feed with bounding box and cosine similarity confidence scores.
  - One-click registration from active camera or external image files.
  - Dynamic sensitivity / cosine similarity threshold slider.
  - Profile manager with delete controls.

---

## Setup & Dependencies

Install required dependencies:

```bash
pip install -r requirements.txt
```

> **CUDA / GPU Note**:
> Ensure you have compatible NVIDIA CUDA / cuDNN drivers installed for `onnxruntime-gpu`. If CUDA is not detected at runtime, the engine will automatically fall back to CPU execution.

---

## Running the Application

```bash
python main.py
```

### Camera Hardware Connection:
- **USB Webcam**: Enter `0`, `1`, etc. in the source input and click **Start Camera**.
- **Doorbell / RTSP Camera**: Enter your camera stream address, e.g.:
  ```text
  rtsp://admin:password@192.168.1.50:554/h264Preview_01_main
  ```

### Registering Face Profiles:
1. Enter person's name in **Person Name**.
2. Click **Capture from Feed** when the person faces the camera, or **From Image** to load a photo.
3. The profile will appear under **Known Profiles Directory**.

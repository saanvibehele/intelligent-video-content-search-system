# Intelligent Video Content Search System

This system enables **semantic search** within videos — allowing users to locate specific objects or words and jump directly to those points in the video. It enhances accessibility and navigation, with responses returned in under 20 seconds.

---

## Features

- **Fast Navigation**: Jump to relevant timestamps in a video via keywords.
- **Semantic Understanding**: Combines image captioning and object detection.
- **Efficient Search**: Natural Language Processing (NLP) + semantic vector search.

---

## Tech Stack

| Tool / Library | Purpose |
|----------------|---------|
| Flask          | API backend for querying and serving responses |
| YOLO (Ultralytics) | Object detection |
| BLIP (via `transformers`) | Scene captioning |
| spaCy          | NLP preprocessing |
| TF-IDF + LSI (via `gensim`) | Semantic search |
| OpenCV         | Frame extraction |
| PyTorch        | Model hosting |

---

## 📁 Folder Structure

```
.
├── app.py               # Flask app
├── process_video.py     # Core processing logic
├── Dockerfile           # Optional: containerize the app
├── requirements.txt     # Python dependencies
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/saanvibehele/intelligent-video-content-search-system.git
cd intelligent-video-content-search-system
```

### 2. Set Up a Python Environment

```bash
python3 -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Download YOLO Model Weights

You need to download a pretrained YOLOv8 model file (`yolov8s-world.pt`) from Ultralytics:

```bash
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-world.pt
```

Place the `yolov8s-world.pt` file in your project directory.

---

## Run the App

```bash
python app.py
```

Visit `http://127.0.0.1:5000` to access the interface or make search queries.

---

## Run with Docker (Optional)

Build and run the Docker container:

```bash
docker build -t intelligent-video-content-search-system .
docker run -p 5000:5000 intelligent-video-content-search-system
```

---

## Notes

- Ensure `ffmpeg` is installed if you're processing videos into frames using OpenCV.

---


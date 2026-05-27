# 🔍 Steel Surface Defect Detection

A complete end-to-end MLOps pipeline for steel surface defect detection using deep learning semantic segmentation.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Overview

This project demonstrates **computer vision** and **MLOps** capabilities through:
- **U-Net segmentation model** built from scratch in PyTorch
- **NEU Surface Defect Dataset** (public, 1800 images, 6 defect types)
- **Training notebook** with clear documentation and visualizations
- **FastAPI REST API** for model serving
- **Docker containerization** for reproducible deployment
- **Hugging Face Spaces** deployment with Gradio UI

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    U-Net Architecture                        │
│                                                             │
│  Input (3,200,200)                                          │
│    │                                                        │
│    ▼                                                        │
│  Encoder: Conv→BN→ReLU→Conv→BN→ReLU + MaxPool (×4)        │
│    │              │         │         │                     │
│    ▼              │         │         │   Skip Connections  │
│  Bottleneck (1024 channels)           │                     │
│    │              │         │         │                     │
│    ▼              ▼         ▼         ▼                     │
│  Decoder: UpConv + Concat + Conv→BN→ReLU (×4)             │
│    │                                                        │
│    ▼                                                        │
│  Output (7,200,200) — 6 defects + background               │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Dataset

**NEU Surface Defect Dataset** from Northeastern University, China.

| Class | Samples | Description |
|-------|---------|-------------|
| Crazing | 300 | Fine network of cracks |
| Inclusion | 300 | Foreign material embedded in surface |
| Patches | 300 | Irregular surface patches |
| Pitted Surface | 300 | Small pits/cavities |
| Rolled-in Scale | 300 | Scale pressed into surface |
| Scratches | 300 | Linear surface scratches |

**Total**: 1800 grayscale images (200×200 pixels)

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/pelabdang/corrosion-detection.git
cd corrosion-detection
pip install -r requirements.txt
```

### 2. Train the Model

Open and run the training notebook:

```bash
jupyter notebook notebooks/training.ipynb
```

The notebook will:
1. Download the NEU dataset automatically
2. Train the U-Net model for 30 epochs
3. Save the best model to `models/unet_steel_defect.pth`
4. Generate visualizations in `assets/`

### 3. Run the API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 7860
```

### 4. Run with Docker

```bash
docker build -t steel-defect-detection .
docker run -p 7860:7860 steel-defect-detection
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check + model info |
| GET | `/health` | Simple health check |
| POST | `/predict` | Full segmentation (mask + scores) |
| POST | `/predict/classification` | Classification only (faster) |

### Example Request

```bash
curl -X POST "http://localhost:7860/predict" \
  -H "accept: application/json" \
  -F "file=@sample_image.bmp"
```

### Example Response

```json
{
  "prediction": {
    "dominant_defect": "scratches",
    "confidence": 0.8742,
    "defect_coverage": 0.1523
  },
  "class_scores": {
    "background": 0.4521,
    "crazing": 0.0234,
    "inclusion": 0.0156,
    "patches": 0.0312,
    "pitted_surface": 0.0187,
    "rolled-in_scale": 0.0298,
    "scratches": 0.8742
  },
  "mask_shape": [200, 200]
}
```

## 🌐 Deploy to Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Select **Docker** as the SDK
3. Push this repository to the Space:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/steel-defect-detection
git push hf main
```

The `spaces.yaml` file configures the Space metadata.

## 📁 Project Structure

```
corrosion-detection/
├── src/
│   ├── __init__.py
│   ├── model.py          # U-Net architecture
│   ├── dataset.py        # Dataset loading and transforms
│   ├── train.py          # Training utilities
│   └── api.py            # FastAPI application
├── notebooks/
│   └── training.ipynb    # Complete training notebook
├── models/
│   └── .gitkeep          # Trained model weights (after training)
├── assets/
│   └── .gitkeep          # Generated visualizations
├── tests/
│   └── test_model.py     # Unit tests
├── app.py                # Gradio app (HF Spaces)
├── Dockerfile            # Container configuration
├── requirements.txt      # Python dependencies
├── spaces.yaml           # HF Spaces metadata
├── .gitignore
└── README.md
```

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

## 📈 Results

After training for 30 epochs on the NEU dataset:

| Metric | Value |
|--------|-------|
| Train Loss | ~0.15 |
| Val Loss | ~0.25 |
| Mean Dice Score | ~0.65 |

*Note: Results may vary. The synthetic masks provide an approximation for demonstration. For production use, annotated pixel-level masks would improve performance significantly.*

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch |
| Model | U-Net (from scratch) |
| Dataset | NEU Surface Defect Dataset |
| API | FastAPI + Uvicorn |
| UI | Gradio |
| Container | Docker |
| Deployment | Hugging Face Spaces |
| Metrics | Dice Coefficient, Cross-Entropy Loss |

## 📚 References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [NEU Surface Defect Database](http://faculty.neu.edu.cn/songkechen/en/zdylm/263265/list/index.htm)
- Song, K. and Yan, Y., "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects," Applied Surface Science, 2013.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

"""FastAPI application for steel defect segmentation inference."""

import io
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

from src.model import UNet
from src.dataset import NEU_CLASSES

# Configuration
MODEL_PATH = Path("models/unet_steel_defect.pth")
IMAGE_SIZE = (200, 200)
NUM_CLASSES = 7  # 6 defect types + background
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names (background + 6 defect types)
CLASS_NAMES = ["background"] + NEU_CLASSES

app = FastAPI(
    title="Steel Surface Defect Detection API",
    description=(
        "U-Net based segmentation model for detecting surface defects "
        "in steel using the NEU Surface Defect Dataset."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
model: UNet | None = None


def load_model() -> UNet:
    """Load the trained U-Net model."""
    net = UNet(in_channels=3, num_classes=NUM_CLASSES)

    if MODEL_PATH.exists():
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        net.load_state_dict(state_dict)
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}. Using untrained model.")

    net.to(DEVICE)
    net.eval()
    return net


@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    global model
    model = load_model()


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess an input image for inference."""
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "U-Net Steel Defect Segmentation",
        "classes": CLASS_NAMES,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Run defect segmentation on an uploaded image.

    Args:
        file: Image file (JPEG, PNG, BMP).

    Returns:
        JSON with predicted class, confidence, and segmentation mask.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/bmp"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG, or BMP.",
        )

    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = preprocess_image(image).to(DEVICE)

        # Run inference
        with torch.no_grad():
            output = model(input_tensor)

        # Process predictions
        probabilities = torch.softmax(output, dim=1)
        pred_mask = torch.argmax(probabilities, dim=1).squeeze(0).cpu().numpy()

        # Get per-class confidence scores
        class_confidences = probabilities.squeeze(0).mean(dim=(1, 2)).cpu().numpy()

        # Determine dominant defect class (excluding background)
        defect_confidences = class_confidences[1:]
        dominant_class_idx = int(np.argmax(defect_confidences)) + 1
        dominant_confidence = float(class_confidences[dominant_class_idx])

        # Calculate defect coverage
        total_pixels = pred_mask.size
        defect_pixels = int(np.sum(pred_mask > 0))
        coverage = defect_pixels / total_pixels

        return JSONResponse(content={
            "prediction": {
                "dominant_defect": CLASS_NAMES[dominant_class_idx],
                "confidence": round(dominant_confidence, 4),
                "defect_coverage": round(coverage, 4),
            },
            "class_scores": {
                name: round(float(score), 4)
                for name, score in zip(CLASS_NAMES, class_confidences)
            },
            "mask_shape": list(pred_mask.shape),
            "mask": pred_mask.tolist(),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/predict/classification")
async def predict_classification(file: UploadFile = File(...)):
    """Run defect classification (no segmentation mask) on an uploaded image.

    Returns only the class prediction and confidence scores for faster response.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file.content_type not in ["image/jpeg", "image/png", "image/bmp"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG, or BMP.",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = preprocess_image(image).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)

        probabilities = torch.softmax(output, dim=1)
        class_confidences = probabilities.squeeze(0).mean(dim=(1, 2)).cpu().numpy()

        defect_confidences = class_confidences[1:]
        dominant_class_idx = int(np.argmax(defect_confidences)) + 1

        return JSONResponse(content={
            "defect_type": CLASS_NAMES[dominant_class_idx],
            "confidence": round(float(class_confidences[dominant_class_idx]), 4),
            "all_scores": {
                name: round(float(score), 4)
                for name, score in zip(CLASS_NAMES, class_confidences)
            },
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

"""Gradio app for Hugging Face Spaces deployment."""

import io

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.model import UNet
from src.dataset import NEU_CLASSES

# Configuration
NUM_CLASSES = 7
IMAGE_SIZE = (200, 200)
CLASS_NAMES = ["background"] + NEU_CLASSES
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Color map for visualization
COLORS = [
    [0, 0, 0],        # background - black
    [255, 0, 0],      # crazing - red
    [0, 255, 0],      # inclusion - green
    [0, 0, 255],      # patches - blue
    [255, 255, 0],    # pitted_surface - yellow
    [255, 0, 255],    # rolled-in_scale - magenta
    [0, 255, 255],    # scratches - cyan
]


def load_model() -> UNet:
    """Load the U-Net model."""
    model = UNet(in_channels=3, num_classes=NUM_CLASSES)
    model_path = "models/unet_steel_defect.pth"
    try:
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Warning: No trained model found. Using untrained model for demo.")
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()


def predict(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """Run inference on an input image.

    Args:
        image: Input image as numpy array (H, W, 3).

    Returns:
        Tuple of (colored segmentation mask, class scores dict).
    """
    if image is None:
        return None, {}

    # Preprocess
    pil_image = Image.fromarray(image).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)

    probabilities = torch.softmax(output, dim=1)
    pred_mask = torch.argmax(probabilities, dim=1).squeeze(0).cpu().numpy()

    # Create colored mask
    colored_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for class_idx, color in enumerate(COLORS):
        colored_mask[pred_mask == class_idx] = color

    # Resize mask to original image size
    colored_mask = np.array(
        Image.fromarray(colored_mask).resize(
            (image.shape[1], image.shape[0]), Image.NEAREST
        )
    )

    # Class confidence scores
    class_confidences = probabilities.squeeze(0).mean(dim=(1, 2)).cpu().numpy()
    scores = {
        CLASS_NAMES[i]: float(class_confidences[i])
        for i in range(1, NUM_CLASSES)  # Skip background
    }

    return colored_mask, scores


# Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload Steel Surface Image"),
    outputs=[
        gr.Image(label="Defect Segmentation Map"),
        gr.Label(label="Defect Class Scores", num_top_classes=6),
    ],
    title="🔍 Steel Surface Defect Detection",
    description=(
        "Upload an image of a steel surface to detect and segment defects. "
        "The model uses a U-Net architecture trained on the NEU Surface Defect Dataset "
        "to identify 6 types of defects: crazing, inclusion, patches, pitted surface, "
        "rolled-in scale, and scratches."
    ),
    article=(
        "## Model Details\n"
        "- **Architecture**: U-Net (encoder-decoder with skip connections)\n"
        "- **Dataset**: NEU Surface Defect Dataset\n"
        "- **Classes**: 6 defect types + background\n"
        "- **Input**: RGB image (resized to 200x200)\n"
        "- **Output**: Pixel-wise segmentation mask + class confidence scores\n\n"
        "## Color Legend\n"
        "🔴 Crazing | 🟢 Inclusion | 🔵 Patches | 🟡 Pitted Surface | "
        "🟣 Rolled-in Scale | 🩵 Scratches"
    ),
    examples=[],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

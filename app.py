from pathlib import Path
import os

import requests
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from plant_disease import IMAGE_SIZE, make_model


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "plant_model.pth"
MODEL_URL = os.environ.get(
    "PLANT_MODEL_URL",
    "https://github.com/palakaghi123/plant-disease-recognition/"
    "releases/download/v1.0/plant_model.pth",
)


def ensure_model():
    if DEFAULT_MODEL.exists():
        return

    st.info("Downloading the demo model; this may take a moment.")
    temporary_model = DEFAULT_MODEL.with_suffix(".pth.download")
    with requests.get(MODEL_URL, stream=True, timeout=(15, 180)) as response:
        response.raise_for_status()
        with temporary_model.open("wb") as model_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    model_file.write(chunk)
    temporary_model.replace(DEFAULT_MODEL)


@st.cache_resource
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    model = make_model(len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, classes


st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="centered")
st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to classify it with the trained ResNet18 model.")
st.warning(
    "Quick demo model: 56% accuracy on a small held-out sample. "
    "It can misclassify leaves and has not been validated on field photos."
)

try:
    ensure_model()
except requests.RequestException as error:
    st.error(f"Could not download the demo model: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.info("Upload a JPG, JPEG, or PNG image to begin.")
else:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded leaf", use_container_width=True)

    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    model, classes = load_model(DEFAULT_MODEL)
    image_tensor = image_transform(image).unsqueeze(0)

    with torch.no_grad():
        probabilities = torch.softmax(model(image_tensor), dim=1)[0]

    predicted_index = probabilities.argmax().item()
    st.success(f"Prediction: {classes[predicted_index]}")
    st.metric("Confidence", f"{probabilities[predicted_index].item():.1%}")

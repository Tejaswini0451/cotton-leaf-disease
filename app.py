
import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
import json
from PIL import Image
from torchvision import models, transforms
import os

# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="Cotton Leaf Disease Detection",
    page_icon="🌿",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

device = torch.device("cpu")

with open(os.path.join(MODEL_DIR, "classes.json")) as f:
    classes = json.load(f)

# ---------------- CUSTOM CNN ----------------

class CottonCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ---------------- LOAD MODELS ----------------

@st.cache_resource
def load_models():

    cnn = CottonCNN(4)
    ckpt = torch.load(
        MODEL_DIR + "/cotton_cnn.pth",
        map_location=device,
        weights_only=False
    )
    cnn.load_state_dict(ckpt["model_state_dict"])
    cnn.to(device).eval()

    resnet = models.resnet18(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, 4)

    ckpt = torch.load(
        MODEL_DIR + "/cotton_resnet18.pth",
        map_location=device,
        weights_only=False
    )
    resnet.load_state_dict(ckpt["model_state_dict"])
    resnet.to(device).eval()

    vit = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=4
    )

    ckpt = torch.load(
        MODEL_DIR + "/cotton_vit.pth",
        map_location=device,
        weights_only=False
    )
    vit.load_state_dict(ckpt["model_state_dict"])
    vit.to(device).eval()

    return cnn, resnet, vit


# ---------------- IMAGE PREPROCESSING ----------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------- PREDICTION ----------------

def predict(model, image):

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    predicted_index = torch.argmax(probabilities).item()

    return (
        classes[predicted_index],
        probabilities[predicted_index].item() * 100,
        probabilities.cpu().numpy()
    )


# ---------------- STREAMLIT UI ----------------

st.title("🌿 Cotton Leaf Disease Detection System")

st.markdown("""
Upload a cotton leaf image to analyze it using three deep learning models:
**Custom CNN, ResNet18 and Vision Transformer (ViT).**
""")

st.info(
    "This prototype is trained on four cotton-leaf classes. "
    "It is not yet validated to reliably reject non-cotton images. "
    "Upload a clear cotton leaf image for prediction."
)

uploaded_file = st.file_uploader(
    "Upload Cotton Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image")

    if st.button("Analyze Leaf", type="primary"):

        if min(image.size) < 32:
            st.error("Image is too small. Please upload a clearer image.")

        else:
            with st.spinner("Analyzing image using all three models..."):

                cnn, resnet, vit = load_models()

                cnn_result = predict(cnn, image)
                resnet_result = predict(resnet, image)
                vit_result = predict(vit, image)

            st.success("Analysis completed!")

            st.subheader("Model Predictions")

            results = {
                "Custom CNN": cnn_result,
                "ResNet18": resnet_result,
                "Vision Transformer": vit_result
            }

            for model_name, result in results.items():

                disease, confidence, probabilities = result

                st.markdown(f"### {model_name}")
                st.write("Prediction:", disease.replace("_", " ").title())
                st.progress(float(confidence / 100))
                st.write(f"Confidence: {confidence:.2f}%")

            st.divider()

            st.subheader("Final Prediction")

            st.markdown(
                f"### 🌱 {vit_result[0].replace('_', ' ').title()}"
            )

            st.metric(
                "ViT Confidence",
                f"{vit_result[1]:.2f}%"
            )

            st.caption(
                "ViT is highlighted because it achieved the highest "
                "test accuracy in this experiment. This is a model "
                "prediction, not a verified agricultural diagnosis."
            )

            st.subheader("Model Comparison")

            comparison = {
                "Model": ["Custom CNN", "ResNet18", "ViT"],
                "Prediction": [
                    cnn_result[0],
                    resnet_result[0],
                    vit_result[0]
                ],
                "Confidence (%)": [
                    round(cnn_result[1], 2),
                    round(resnet_result[1], 2),
                    round(vit_result[1], 2)
                ]
            }

            st.dataframe(comparison)

st.divider()

st.caption(
    "Academic project | Cotton Leaf Disease Classification "
    "using CNN, ResNet18 and Vision Transformer"
)

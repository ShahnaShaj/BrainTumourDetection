import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn

# Streamlit App
st.title("🧠 Brain Tumor Detection")
with st.expander("ℹ️ About the Model"):
    st.write("This app detects brain tumors from MRI images using deep learning. It was trained on a Kaggle " \
    "Brain MRI dataset using four ResNet models — ResNet-34, 50, 101, and 152. ResNet-50 and ResNet-152 " \
    "performed best and are used here for real-time predictions.")
st.write("Upload an MRI image and see predictions from multiple ResNet models.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

# Pre-load model architectures and paths
model_dict = {
    "Resnet 34": (models.resnet34, r"C:\AbeerPython\resnet34_state_dict (1).pth"),
    "Resnet 50": (models.resnet50, r"C:\AbeerPython\resnet50_brain_tumor.pth"),
    "Resnet 101": (models.resnet101, r"C:\AbeerPython\resnet101_state_dict.pth"),
    "Resnet 152": (models.resnet152, r"C:\AbeerPython\resnet152_mri.pth"),
}

# Cache model loading
@st.cache_resource
def load_model(_arch, path):
    model = _arch(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize to default 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Prediction classes
classes = ["No Tumor", "Tumor"]

# If file is uploaded
if uploaded_file is not None:
    # Resize uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    st.image(image, caption="🖼️ Resized MRI Image (224x224)")

    img_tensor = transform(image).unsqueeze(0)

    st.subheader("🔎 Predictions by Different Models:")

    # Display predictions for all models
    for model_name, (arch, path) in model_dict.items():
        model = load_model(arch, path)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            pred = torch.argmax(probs).item()

        st.markdown(f"### {model_name}: **{classes[pred]}**")
        st.progress(float(probs[pred]))
        st.write(f"Confidence: **{probs[pred]:.4f}**")
    st.divider()

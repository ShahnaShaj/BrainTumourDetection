# app.py
import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image

# Title
st.title("Brain MRI Tumor Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "png", "jpeg"])

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load model
@st.cache(allow_output_mutation=True)
def load_model():
    model = models.resnet152(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)  # Adjust to your number of classes
    model.load_state_dict(torch.load("resnet152_mri_full.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Predict
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_column_width=True)
    
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        class_names = ['No Tumor', 'Glioma Tumor', 'Meningioma Tumor']  # Adjust to your classes
        prediction = class_names[predicted.item()]
        st.success(f"Prediction: **{prediction}**")

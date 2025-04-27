# 🧠 Brain Tumor Detection using Deep Learning

## 📋 Overview
This project focuses on detecting brain tumors from MRI images using deep learning models, primarily various ResNet architectures (ResNet-34, ResNet-50, ResNet-101, ResNet-152).  
The goal was to build an accurate, efficient classification model to distinguish between **tumorous** and **non-tumorous** brain scans.

## ⚙️ Tech Stack
- **Deep Learning Framework**: [PyTorch](https://pytorch.org/)
- **Pre-trained Model**: [ResNet](https://pytorch.org/vision/stable/models.html#id6)
- **Data Augmentation**: [TorchVision](https://pytorch.org/vision/stable/transforms.html)
- **Visualization**: [Matplotlib](https://matplotlib.org/)
- **Model Saving/Loading**: [PyTorch](https://pytorch.org/docs/stable/torch.html#torch.save)
- **IDE**: Jupyter Notebook

## 🏗️ Model Architectures and Contributors
- **ResNet-34: Tehzeeb https://drive.google.com/file/d/1tss5qsg3hhXTJPbVEVYoPsuRy5m2WKSp/view?usp=drive_link**
- **ResNet-50: Abeer https://drive.google.com/file/d/1vUSWxY-cF4PqrLd9r43EZ9wWyL4mP317/view?usp=drive_link**
- **ResNet-101: Palika https://drive.google.com/file/d/1BMuGxVBE2NhlmmzbFt6XOcfztYTwM2Wq/view?usp=drive_link**
- **ResNet-152: Shahna https://drive.google.com/file/d/1UTZacGtd21cXD7DRB1GrBe8mIyJP0hXi/view?usp=drive_link**

Each model was:
- Modified to output **2 classes**: **tumor** and **no tumor**.
- **Fine-tuned** on pre-trained **ImageNet** weights.
- Trained using **CrossEntropy Loss** and **Adam optimizer**.
## 🚀 How to Run  
1️⃣ Install dependencies:  
```bash
pip install -r requirements.txt
```
2️⃣  Run the Streamlit app:
```bash
streamlit run app.py
```

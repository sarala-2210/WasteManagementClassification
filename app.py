import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
import gdown
import os

# Page setup
st.set_page_config(
    page_title="♻️ Waste Classification App",
    page_icon="🧠",
    layout="centered"
)

# Paths and model download
model1_path = "best_model.keras"
model2_file_id = "19DFx7QKHTJgjss6qcNedZYqoHaf1pXqD"
model2_path = "model2.keras"

def download_model_from_drive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

# Download model2 from Google Drive if not already present
download_model_from_drive(model2_file_id, model2_path)

# Load models
model1 = load_model(model1_path)
model2 = load_model(model2_path)

# Target image size expected by the models
TARGET_SIZE = (224, 224)

# Class labels
categories = {
    0: 'paper', 1: 'cardboard', 2: 'plastic', 3: 'metal', 4: 'trash', 5: 'battery',
    6: 'shoes', 7: 'clothes', 8: 'green-glass', 9: 'brown-glass', 10: 'white-glass',
    11: 'biological'
}

# Image preprocessing
val_test_datagen = ImageDataGenerator(rescale=1. / 255)

def preprocess_single_image(img):
    img = img.resize(TARGET_SIZE)
    img_array = img_to_array(img)
    img_array = img_array.reshape((1,) + img_array.shape)
    img_array = val_test_datagen.standardize(img_array)
    return img_array

# Streamlit UI
st.title("🧠 Waste Classification using CNN")
st.write("Upload a waste image to see predictions from both models.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    input_image = preprocess_single_image(img)

    # Predictions from both models
    prediction_1 = model1.predict(input_image)
    prediction_2 = model2.predict(input_image)

    # Extract predicted classes and confidence
    class_1 = categories[np.argmax(prediction_1[0])]
    confidence_1 = np.max(prediction_1[0]) * 100

    class_2 = categories[np.argmax(prediction_2[0])]
    confidence_2 = np.max(prediction_2[0]) * 100

    # Display predictions
    st.markdown("### 🔎 Prediction Results")
    st.markdown(f"**Model 1** — Class: `{class_1.upper()}` | Confidence: `{confidence_1:.2f}%`")
    st.markdown(f"**Model 2** — Class: `{class_2.upper()}` | Confidence: `{confidence_2:.2f}%`")
else:
    st.info("👆 Upload an image to get predictions.")

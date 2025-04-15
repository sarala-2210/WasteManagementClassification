import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

# Load your best saved model
model = load_model("best_model.keras")

# Define target image size (should match your model input)
TARGET_SIZE = (224, 224)

# Replace with actual class names in order of training labels
# Dictionary to save our 12 classes
categories = {0: 'paper', 1: 'cardboard', 2: 'plastic', 3: 'metal', 4: 'trash', 5: 'battery',
              6: 'shoes', 7: 'clothes', 8: 'green-glass', 9: 'brown-glass', 10: 'white-glass',
              11: 'biological'}

# ImageDataGenerator for test-time preprocessing (same as val/test generator)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Preprocess a single image using same settings as test_generator
def preprocess_single_image(img):
    img = img.resize(TARGET_SIZE)
    img_array = img_to_array(img)
    img_array = img_array.reshape((1,) + img_array.shape)  # Add batch dimension
    img_array = val_test_datagen.standardize(img_array)    # Apply rescaling
    return img_array

# Streamlit UI
st.title("🧠 Deep Learning Image Classifier")
st.write("Upload an image and I'll tell you what it is!")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    input_image = preprocess_single_image(img)
    prediction = model.predict(input_image)

    predicted_class = categories[np.argmax(prediction[0])]
    confidence = np.max(prediction[0]) * 100

    st.markdown(f"### 🏷️ Prediction: **{predicted_class}**")
    st.markdown(f"**Confidence:** {confidence:.2f}%")

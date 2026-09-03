import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

@tf.keras.utils.register_keras_serializable()
def preprocess_mobilenet(x):
    return tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_tea_model.keras", safe_mode=False)
    with open("class_names.json") as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model()

IMG_HEIGHT = 224
IMG_WIDTH = 224

# ---------------------------
# UI
# ---------------------------
st.title("🍃 Tea Leaf Disease Classifier")
st.write("Upload a tea leaf image to detect disease type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("Analyzing..."):
        predictions = model.predict(img_array)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx] * 100

    st.subheader("Prediction")
    st.write(f"**{class_names[predicted_idx]}**")
    st.write(f"Confidence: {confidence:.2f}%")

    # Show all class probabilities
    st.subheader("All class probabilities")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {predictions[0][i]*100:.2f}%")
        st.progress(float(predictions[0][i]))
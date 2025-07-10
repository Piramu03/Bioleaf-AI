import streamlit as st # type: ignore
from PIL import Image
import json
from tensorflow.keras.models import load_model  # type: ignore
from utils.preprocessor import preprocess_image
from utils.disease_info import DISEASE_INFO
import os
import numpy as np

# ✅ Page config must be first Streamlit call
st.set_page_config(page_title="Plant Disease Detector", layout="wide")

# ✅ Custom CSS to reduce file uploader size
st.markdown(
    """
    <style>
    div[data-testid="stFileUploaderDropzone"] {
        padding: 6px;
        height: 65px;
        border-radius: 8px;
    }
    div[data-testid="stFileUploaderDropzone"] p {
        font-size: 13px;
        margin-top: 0px;
    }
    div[data-testid="stFileUploader"] button {
        padding: 4px 12px;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Suppress TensorFlow & oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def load_model_classes():
    try:
        model = load_model('models/best_model.h5')
        with open('models/class_indices.json', 'r') as f:
            class_indices = json.load(f)
        class_names = {v: k for k, v in class_indices.items()}
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def predict_disease(image, model, class_names, confidence_threshold=0.7):
    try:
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names[predicted_class_index]
        confidence = float(prediction[0][predicted_class_index])

        if confidence < confidence_threshold:
            return "Unknown", confidence
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Error", 0.0

def display_result(image, disease, confidence):
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    with col2:
        if disease == "Unknown":
            st.warning("Could not confidently detect the disease. Try another image.")
            st.info(f"Confidence: {confidence:.2%}")
        elif disease == "Error":
            st.error("⚠️ Prediction failed due to an internal error.")
        else:
            st.success(f"✅ Disease Detected: {disease.replace('_', ' ')}")
            st.progress(confidence)
            st.info(f"Confidence: {confidence:.2%}")

            st.subheader("Disease Information")
            st.write(f"**Disease Name:** {disease.replace('_', ' ')}")

            if disease in DISEASE_INFO:
                info = DISEASE_INFO[disease]

                with st.expander("Cause"):
                    st.write(info['cause'])

                with st.expander("Symptoms"):
                    for symptom in info['symptoms']:
                        st.write(f"• {symptom}")

                with st.expander("Treatment Recommendations"):
                    for treatment in info['treatment']:
                        st.write(f"• {treatment}")

                with st.expander("Prevention Measures"):
                    for prevention in info['prevention']:
                        st.write(f"• {prevention}")

                with st.expander("🛍️ Recommended Products"):
                    for product in info.get('products', []):
                        st.write(f"• {product}")
            else:
                st.warning("ℹ️ Detailed info not available for this disease.")

def main():
    # 🌿 App Title
    st.markdown(
        "<h1 style='text-align: center;'>🌿 Plant Disease Detection System 🌿</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center;'>Upload leaf images or use the camera to detect plant diseases and get remedies.</p>",
        unsafe_allow_html=True
    )

    model, class_names = load_model_classes()
    if not model or not class_names:
        st.stop()

    # Tabs for upload or camera
    tab1, tab2 = st.tabs(["📁 Upload Images", "📸 Use Camera"])

    with tab1:
        uploaded_files = st.file_uploader(
            "Upload one or more plant leaf images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        if uploaded_files and st.button("Analyze Uploaded Images"):
            with st.spinner("Analyzing uploaded images..."):
                for uploaded_file in uploaded_files:
                    image = Image.open(uploaded_file)
                    disease, confidence = predict_disease(image, model, class_names)
                    st.markdown("---")
                    display_result(image, disease, confidence)

    with tab2:
        image_data = st.camera_input("Capture a leaf image using your webcam")

        if image_data is not None and st.button("📷 Analyze Captured Image"):
            with st.spinner("Analyzing captured image..."):
                image = Image.open(image_data)
                disease, confidence = predict_disease(image, model, class_names)
                st.markdown("---")
                display_result(image, disease, confidence)

if __name__ == '__main__':
    main()

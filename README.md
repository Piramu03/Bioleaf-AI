# BioLeaf AI – Plant Disease Detection System

BioLeaf AI is a deep learning-powered web application designed to identify plant diseases from leaf images. It provides accurate predictions along with treatment and prevention advice. Built with TensorFlow, Keras (MobileNetV2), and Streamlit, this tool aims to assist farmers, students, and agricultural researchers.

---

## 📝 Description

BioLeaf AI allows users to upload or capture images of plant leaves. A deep learning model analyzes the image and predicts the disease (if present), then displays information including symptoms, causes, treatments, and recommended agricultural products.

This project uses a pre-trained MobileNetV2 model, trained on the PlantVillage dataset.

---

## 🚀 Features

- Upload one or more plant leaf images
- Use live webcam capture for real-time analysis
- Predict disease with confidence score
- View symptoms, causes, treatments, and prevention
- Get agricultural product recommendations
- Fast and responsive Streamlit web interface
- Lightweight MobileNetV2 model (optimized for performance)

---
## 1.Install Python Dependencies
pip install -r requirements.txt

## 2.Run the Application
streamlit run app.py

---
json

## 📁 Uploading Images
There are two ways to provide leaf images to the system:

Option 1: Upload Images
Drag and drop multiple .jpg, .jpeg, or .png files

Click "Analyze Uploaded Images" to view predictions

Option 2: Use Camera
Click "Capture a leaf image using your webcam"

Then click "Analyze Captured Image"

The system will:

Display the uploaded image

Predict the disease

Show confidence score

Provide recommendations

## 🌿 Disease Information
After the prediction, the app displays detailed info:

Disease Name – e.g., Tomato Leaf Curl Virus

Cause – e.g., Viral infection by whiteflies

Symptoms – e.g., Yellowing, curling of leaves

Treatment – e.g., Use insecticides, remove infected plants

Prevention – e.g., Use resistant varieties, rotate crops

Recommended Products – e.g., Neem oil, Biocontrol agents


## 📄 License
This project is licensed under the MIT License.
You may use, modify, and distribute it for academic, research, or educational purposes.

---

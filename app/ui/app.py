import streamlit as st
import requests
from PIL import Image
import pandas as pd
import os

# Config
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Image Classifier", page_icon="🔍")

st.title("🔍 Pneumonia Image Classifier")

# File upload
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Predict button
    if st.button("Classify Image"):
        with st.spinner("Predicting..."):
            # Call API
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(f"{API_URL}/predict", files=files)
            
            if response.status_code == 200:
                result = response.json()
                
                # Show results
                st.success(f"Predicted: **{result['predicted_class']}**")
                st.write(f"Confidence: {result['confidence']:.2%}") 
            else:
                st.error("Prediction failed!")
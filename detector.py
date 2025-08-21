import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import zipfile
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Cassava Mosaic Disease Detector",
    page_icon="🌿",
    layout="wide"
)

class CassavaModel:
    def __init__(self):
        self.model = None
        self.history = None
        self.img_size = (224, 224)
        self.batch_size = 32
           
    def predict_image(self, image):
        """Predict disease from image"""
        if isinstance(image, np.ndarray):
            img = cv2.resize(image, self.img_size)
        else:
            img = image.resize(self.img_size)
            img = np.array(img)
        
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        prediction = self.model.predict(img, verbose=0)[0][0]
        
        if prediction > 0.5:
            return "Infected", prediction
        else:
            return "Healthy", 1 - prediction

def main():
    st.title("🌿 Cassava Mosaic Disease Detection System")
    st.markdown("### Deep Learning Model for Early Detection of Cassava Mosaic Diseases")
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = CassavaModel()
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    
    model = st.session_state.model
    
    st.markdown("""
    ## About Cassava Mosaic Disease
    
    Cassava Mosaic Disease (CMD) is one of the most devastating diseases affecting cassava crops in Africa.
    This application uses deep learning to detect early signs of CMD in cassava leaves.
    """)

    st.header("Disease Detection")
    
    # Display sample images if available
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Healthy Cassava Leaf")
        st.image("healthy.jpg", 
                caption="Example of healthy cassava leaf")
    
    with col2:
        st.subheader("Infected Cassava Leaf")
        st.image("infected.jpg", 
                caption="Example of CMD infected leaf")
    MODEL_PATH = "cassava_model.h5"

    # Check if model exists, if not -> download
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            url = "https://drive.google.com/uc?id=YOUR_FILE_ID"  # <-- replace with your file id
            gdown.download(url, MODEL_PATH, quiet=False)
    
    if st.session_state.trained or os.path.exists(MODEL_PATH):
        if not st.session_state.trained:
            with st.spinner("Loading saved model..."):
                model.model = tf.keras.models.load_model(MODEL_PATH)
                st.session_state.trained = True

        # Option to choose input method
        input_method = st.radio(
            "Select input method:",
            ("Upload Image", "Use Camera")
        )

        image = None

        if input_method == "Upload Image":
            uploaded_image = st.file_uploader("Upload a cassava leaf image", 
                                            type=['jpg', 'jpeg', 'png'])
            if uploaded_image:
                image = Image.open(uploaded_image)

        elif input_method == "Use Camera":
            camera_image = st.camera_input("Take a picture")
            if camera_image:
                image = Image.open(camera_image)

        if image:
            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Selected Image", use_column_width=True)

            with col2:
                with st.spinner("Analyzing image..."):
                    result, confidence = model.predict_image(image)

                    if result == "Infected":
                        st.error(f"⚠️ **Disease Detected!**")
                        st.error(f"Confidence: {confidence:.2%}")
                        st.markdown("""
                        **Recommendations:**
                        - Remove infected plants immediately  
                        - Apply appropriate fungicide treatment  
                        - Monitor surrounding plants closely  
                        """)
                    else:
                        st.success(f"✅ **Healthy Plant!**")
                        st.success(f"Confidence: {confidence:.2%}")
                        st.markdown("""
                        **Recommendations:**
                        - Continue regular monitoring  
                        - Maintain good plant hygiene  
                        - Apply preventive measures  
                        """)

        # Batch prediction option (upload only)
        st.subheader("Batch Prediction")
        uploaded_files = st.file_uploader("Upload multiple images", 
                                        type=['jpg', 'jpeg', 'png'], 
                                        accept_multiple_files=True)

        if uploaded_files:
            results = []
            for i, file in enumerate(uploaded_files):
                image = Image.open(file)
                result, confidence = model.predict_image(image)
                results.append({
                    'Image': file.name,
                    'Prediction': result,
                    'Confidence': f"{confidence:.2%}"
                })

            df_results = pd.DataFrame(results)
            st.dataframe(df_results)

            # Summary statistics
            infected_count = len(df_results[df_results['Prediction'] == 'Infected'])
            healthy_count = len(df_results[df_results['Prediction'] == 'Healthy'])

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Images", len(df_results))
            col2.metric("Infected", infected_count)
            col3.metric("Healthy", healthy_count)

    else:
        st.warning("Please train the model first or ensure a trained model exists.")
if __name__ == "__main__":
    main()
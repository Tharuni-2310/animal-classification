import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import pickle
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2

# Load your trained model
clf = pickle.load(open("model(1).pkl", "rb"))
categories = ['Dog', 'Cat', 'Cow', 'Monkey', 'Elephant']

# Load MobileNetV2 model once
feature_extractor = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

st.title("🐾 Animal Image Classifier")
st.write("Paste an image URL and the model will predict which animal it is.")

url = st.text_input("Enter Image URL here")

if url:
    try:
        # Fetch and display image
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        st.image(img, caption="Input Image", use_column_width=True)

        # Preprocess image for feature extraction
        img = img.resize((150, 150))
        x = np.array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Extract features and predict
        features = feature_extractor.predict(x).flatten().reshape(1, -1)
        pred_class = clf.predict(features)[0]

        st.success(f"Predicted Animal: **{categories[pred_class]}**")

    except Exception as e:
        st.error(f"Error processing image: {e}")

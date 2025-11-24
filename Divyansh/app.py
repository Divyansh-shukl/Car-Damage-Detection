import streamlit as st
import requests
from PIL import Image

API_URL = "https://car-damage-detection-fastapi-server.onrender.com/predict"

st.set_page_config(page_title="Car Damage Detection", layout="centered")
st.title("🚗 Vehicle Damage Detection")

uploaded_file = st.file_uploader("Upload the file", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analysing image..."):
        files = {"file": ("upload.jpg", uploaded_file.getbuffer(), uploaded_file.type)}
        try:
            response = requests.post(API_URL, files=files)
            result = response.json()

            if "prediction" in result:
                st.success(f"Predicted Class: **{result['prediction']}**")
            else:
                st.error(f"Error: {result}")

        except Exception as e:
            st.error(f"Request failed: {e}")

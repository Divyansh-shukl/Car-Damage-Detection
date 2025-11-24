# Divyansh/app.py
import streamlit as st
import requests
from PIL import Image

API_URL = "https://car-damage-detection-fastapi-server.onrender.com/predict"

st.set_page_config(page_title="Car Damage Detection", layout="centered")
st.title("🚗 Vehicle Damage Detection (API)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # show preview
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # prepare request (preserve original filename/type)
    files = {"file": (uploaded_file.name, uploaded_file.getbuffer(), uploaded_file.type or "image/jpeg")}

    with st.spinner("Sending image to the model..."):
        try:
            resp = requests.post(API_URL, files=files, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if "prediction" in data:
                st.success(f"Predicted class: **{data['prediction']}**")
            else:
                st.error(f"Unexpected response: {data}")
        except Exception as e:
            st.error(f"Request failed: {e}")
else:
    st.info("Upload an image to get a prediction from the API.")

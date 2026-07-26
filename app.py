import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page settings
st.set_page_config(page_title="Fruit Object Detection", layout="wide")

st.title("🍎 Fruit Detection using YOLOv8")
st.write("Upload an image and detect fruits.")

# Load model
model = YOLO("best.pt")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_array = np.array(image)

    with st.spinner("Detecting objects..."):
        results = model.predict(
            source=image_array,
            conf=0.25
        )

    result = results[0]

    annotated = result.plot()

    st.subheader("Detection Result")

    st.image(annotated, use_container_width=True)

    st.subheader("Detected Objects")

    if len(result.boxes) == 0:
        st.warning("No objects detected.")
    else:
        for box in result.boxes:
            cls = int(box.cls)
            conf = float(box.conf)

            st.write(
                f"**{model.names[cls]}** — {conf:.2%}"
            )
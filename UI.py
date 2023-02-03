import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tensorflow import keras
import cv2
import numpy as np
import os
from pathlib import Path
from numpy import asarray

# st.header("EXPLORE BEYOND :white[colors]")

st.markdown(
    f'<h1 style="color:#FFFFFF;font-size:36px;">{"EXPLORE BEYOND"}</h1>',
    unsafe_allow_html=True,
)
# st.sidebar("")
# img = Image.open("mars.jpg")

# st.image(img, width=1000)


def add_bg_from_url():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background-image: url("https://images.hdqwalls.com/download/mars-planet-view-4k-rn-1920x1080.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True,
    )


add_bg_from_url()

upload = st.file_uploader("Upload your satellite image", type=["jpg", "png", "jpeg"])

# save_folder = (
#     r"C:\Users\raghu\OneDrive\Documents\Projects\Explore-Beyond\uploaded images"
# )
# save_path = Path(save_folder)
with open(os.path.join("uploaded images", upload.name), "wb") as f:
    f.write(upload.getbuffer())

st.success(f"File {upload.name} is successfully saved!")

model = keras.models.load_model(r"./models/cnn.h5")
CATEGORIES = ["bright_dune", "crater", "dark_dune", "impact_ejecta", "slope_streak"]


def findClass(upload):
    image_1 = cv2.imread(upload)
    image_1 = cv2.resize(upload, (227, 227))
    np.array(image_1).reshape(-1, 227, 227, 3)
    image_1 = image_1.astype("float32")
    image_1 /= 255
    prediction = model.predict(np.array([image_1], np.float32))
    # print(prediction)
    return CATEGORIES[np.argmax(prediction)]


if upload is not None:
    image = Image.open(upload)
    # numpydata = asarray(image)
    # findClass(numpydata)
    fig = plt.figure()
    plt.imshow(image)
    plt.axis("off")
    st.pyplot(fig)
    # result = predict_class(image)

# image_path = f"uploaded images/{upload.name}.jpg"


# findClass(upload)

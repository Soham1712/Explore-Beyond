import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tensorflow import keras
import cv2
import numpy as np
import os
from pathlib import Path
from numpy import asarray
from PIL import Image

# st.header("EXPLORE BEYOND :white[colors]")

image = Image.open("mars.jpg")

st.sidebar.image(image, width=300)
rad = st.sidebar.radio("", ["Prediction","Statistics", "About"])


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


if rad == "Prediction":
    st.markdown(
        f'<h1 style="color:#FFFFFF;font-size:36px;">{"EXPLORE BEYOND"}</h1>',
        unsafe_allow_html=True,
    )

    add_bg_from_url()

    st.header("Upload Your Satellite Image")

    upload = st.file_uploader(" ", type=["jpg", "png", "jpeg"])

    if(upload):
        with open(os.path.join("uploaded_file", "test"), "wb") as f:
            f.write(upload.getbuffer())

        st.success(f"File {upload.name} is successfully saved!")

    CATEGORIES = ["bright_dune", "crater", "dark_dune", "impact_ejecta", "slope_streak"]

    model_cnn = keras.models.load_model(r"./models/cnn.h5")
    model_alexnet = keras.models.load_model(r"./models/cnn3.h5")

    def findClass1(upload):
        image_1 = cv2.imread(upload)
        image_1 = cv2.resize(image_1, (227, 227))
        np.array(image_1).reshape(-1, 227, 227, 3)
        image_1 = image_1.astype("float32")
        image_1 /= 255
        prediction = model_cnn.predict(np.array([image_1], np.float32))
        # print(prediction)
        return CATEGORIES[np.argmax(prediction)]

    def findClass2(upload):
        image_1 = cv2.imread(upload)
        image_1 = cv2.resize(image_1, (227, 227))
        np.array(image_1).reshape(-1, 227, 227, 3)
        image_1 = image_1.astype("float32")
        image_1 /= 255
        prediction = model_alexnet.predict(np.array([image_1], np.float32))
        # print(prediction)
        return CATEGORIES[np.argmax(prediction)]

    if upload is not None:
        image = Image.open(upload)
        # numpydata = asarray(image)
        result_cnn = findClass1("uploaded_file/test")
        result_alexnet = findClass2("uploaded_file/test")
        fig = plt.figure()
        plt.imshow(image)
        plt.axis("off")
        st.pyplot(fig)
        st.header("CNN")
        st.subheader(result_cnn)
        st.header("Alexnet")
        st.subheader(result_alexnet)

if rad == "Statistics":
    st.header("Statistics")
    add_bg_from_url()


if rad == "Infographics":
    st.header("Graphs")


if rad == "About":
    st.header("About")
    add_bg_from_url()
    

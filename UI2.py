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
# import mysql.connector

# st.header("EXPLORE BEYOND :white[colors]")

image_sidebar = Image.open(r"./Images/contents.png")
image_crater = Image.open(r"./Images/Crater.jpg")
image_darkDune = Image.open(r"./Images/DarkDune.jpg")
image_brightDune = Image.open(r"./Images/BrightDune.jpg")
image_impactEjecta = Image.open(r"./Images/ImpactEjecta.jpg")
image_slopeStreak = Image.open(r"./Images/SlopeStreak.jpg")

st.sidebar.image(image_sidebar, width=300)
rad = st.sidebar.radio("", ["Prediction", "About"])


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


model_cnn = keras.models.load_model(r"./models/cnn.h5")
model_alexnet = keras.models.load_model(r"./models/cnn3.h5")


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

    CATEGORIES = ["Bright Dune", "Crater", "Dark Dune", "Impact Ejecta", "Slope Streak"]

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

if rad == "Analytics":
    st.header("Analytics")


if rad == "Infographics":
    st.header("Graphs")


if rad == "About":
    st.header("Planetary Structures ")

    st.subheader("Crater")
    col1, col2 = st.columns(2)
    col1.image(image_crater, width=200)
    col2.markdown(
        '<div style="text-align: justify;">A large bowl-shaped cavity in the ground or on a celestial object, typically one caused by an explosion or the impact of a meteorite. The counting and size of craters provide us with the planets and moons absolute and relative ages. </div>',
        unsafe_allow_html=True,
    )

    st.subheader("Dark Dune")
    col3, col4 = st.columns(2)
    col3.image(image_darkDune, width=200)
    col4.markdown(
        '<div style="text-align: justify;">A dune is a heap of sand piled up and shaped by the wind. The dark dunes are composed of basaltic sand, and scientists believe the dunes in the image above have formed in response to fall and winter westerly winds. Also superimposed on their surface are smaller secondary dunes that are commonly seen on terrestrial dunes of this size. </div>',
        unsafe_allow_html=True,
    )

    st.subheader("Bright Dune")
    col5, col6 = st.columns(2)
    col5.image(image_brightDune, width=200)
    col6.markdown(
        '<div style="text-align: justify;">A dune is a heap of sand piled up and shaped by the wind. The bright patches are made up of large ridges that look like wind-blown bedforms. Additionally, the bright patches are yellowish in the infrared-red-blue image. In enhanced color, most sand on Mars is blue but dust is yellow. This suggests that the bright bedforms are either built from, or covered by, dust or material with a different composition. </div>',
        unsafe_allow_html=True,
    )

    st.subheader("Slope Streak")
    col7, col8 = st.columns(2)
    col7.image(image_slopeStreak, width=200)
    col8.markdown(
        '<div style="text-align: justify;">Elongated streak-like features on slopes. Slope streaks have sharp edges and downward increasing width. They form in relatively steep terrain, such as along escarpments and crater walls. </div>',
        unsafe_allow_html=True,
    )

    st.subheader("Impact Ejecta")
    col9, col10 = st.columns(2)
    col9.image(image_impactEjecta, width=200)
    col10.markdown(
        '<div style="text-align: justify;">Impact ejecta is material excavated from a crater cavity during impact; most ejecta (when ejecta velocity is less than escape velocity) will form a deposit, or layer, of debris surrounding the crater cavity, thinning with distance. </div>',
        unsafe_allow_html=True,
    )

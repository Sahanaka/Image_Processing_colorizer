'''
    EG/2017/3059 - Herath H.M.D.P.M
    EG/2017/3079 - Weraniyagoda W.A.S.A

    Image Processing colorizer

    This model is trained to identify scenery and also humans to a detailed level and perform colorization on them.
'''

import numpy as np
import cv2
import streamlit as st
from PIL import Image

# Main function to convert the image to colorized image


def colorizer(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    prototxt = r"./models/models_colorization_deploy_v2.prototxt"
    model = r"./models/colorization_release_v2.caffemodel"
    points = r"./models/pts_in_hull.npy"
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)

    colorized = (255 * colorized).astype("uint8")

    # Return the colorized images

    return colorized

# UI for the app


st.write("""
          # Gray scale to Color Image Converter
          """
         )

st.write("This app can Colorize a given Gray scale image.")
st.write("Contributors: Herath H.M.D.P.M (EG/2017/3059) & Weraniyagoda W.A.S.A (EG/2017/3079)")

file = st.sidebar.file_uploader(
    "Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    img = np.array(image)

    st.text("Uploaded Image")
    st.image(image, use_column_width=True)

    st.text("Colorized Image")
    color = colorizer(img)

    st.image(color, use_column_width=True)

    im = Image.fromarray(color)
    im.save("filename.jpeg")
    print("Done!")

    # Download Button

    with open("filename.jpeg", "rb") as file:
        btn = st.download_button(
            label="Download image",
            data=file,
            file_name="download.png",
            mime="image/png"
        )
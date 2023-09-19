import streamlit as st
from keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from keras.preprocessing import image
from keras import backend as K
import numpy as np
from PIL import Image
import pandas as pd
import tensorflow as tf
import cv2

# prepare learned CNN model
model = VGG16(weights="imagenet", include_top=True)
model.summary()


# make heatmap by grad-cam
def make_heatmap(last_conv_layer_name, model, target_image):
    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer(last_conv_layer_name)
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(target_image)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    heatmap_shape = (grads.shape[1], grads.shape[2])
    heatmap_emphasis = np.maximum(heatmap, 0)
    heatmap_emphasis /= np.max(heatmap_emphasis)
    heatmap_emphasis = heatmap_emphasis.reshape(heatmap_shape)
    return heatmap_emphasis


# title
st.title("Image Recognition")

# abstract
st.write("When you input an image, the result of image recognition by CNN and the judgment part are shown by grad-CAM.")

# image input
uploaded_file = st.file_uploader("File Upload", type='jpg')
if uploaded_file:
    input_img = Image.open(uploaded_file).resize((224, 224))

    # prepare img
    preprocessed_img = np.stack([image.img_to_array(input_img)])

    # predict
    results = decode_predictions(model.predict(preprocessed_img), top=10)

    # show image
    st.write("image")
    st.image(input_img, width=400)

    # show bar chart
    st.write("recognition result (%)")
    data = pd.DataFrame(
        [row[2] * 100 for row in results[0]],
        index=[row[1] for row in results[0]]
    )
    st.bar_chart(data)

    heatmap = make_heatmap('block5_conv3', model, preprocess_input(preprocessed_img))

    cv_img = cv2.cvtColor(np.array(input_img), cv2.COLOR_RGB2BGR)

    INTENSITY = 0.6

    heatmap = cv2.resize(heatmap, (cv_img.shape[1], cv_img.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    img = heatmap * INTENSITY + cv_img

    cv2.imwrite('heatmap.jpg', img)

    # show judgement part
    st.write("judgement part")

    st.image('heatmap.jpg', width=400)

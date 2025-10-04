import os
import re
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.applications import efficientnet
from keras.layers import TextVectorization
from keras.preprocessing.image import load_img, img_to_array
import streamlit as st

# Constants from the notebook
IMAGE_SIZE = (299, 299)
VOCAB_SIZE = 13000
SEQ_LENGTH = 24
EMBED_DIM = 512
FF_DIM = 512


MODEL_PATH = 'model.keras'  

# Custom standardization function from the notebook
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    strip_chars = "!\"#$%&'()*+,-./:;=?@[\\]^_`{|}~1234567890"
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")


vectorization = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQ_LENGTH,
    standardize=custom_standardization
)


# Image decoding and resizing function from the notebook
def decode_and_resize(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

# Inference function (adapted from the notebook's get_caption)
def generate_caption(model, img_path, vectorization):
    img = decode_and_resize(img_path)
    img = tf.expand_dims(img, 0)  # Batch dimension
    
    # Initial caption: <start>
    caption = tf.constant(["<start>"])
    caption = vectorization(caption)
    
    for i in range(SEQ_LENGTH - 1):
        predictions = model.predict([img, caption], verbose=0)
        predicted_id = tf.argmax(predictions[:, i, :], axis=-1)
        predicted_word = vectorization.get_vocabulary()[predicted_id[0].numpy()]
        
        if predicted_word == "<end>":
            break
        
        # Append the predicted word
        caption = tf.concat([caption, tf.expand_dims(predicted_id, 0)], axis=-1)
    
    # Decode the caption
    caption_text = " ".join([vectorization.get_vocabulary()[token] for token in caption[0].numpy() if token != 0][1:-1])  # Remove <start> and <end>
    return caption_text.capitalize()

# Load the model
@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

# Streamlit app
st.title("Image Captioning with Transformers")

# File uploader
uploaded_file = st.file_uploader("Upload an image (JPG)", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    img_path = os.path.join("temp.jpg")
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the image
    st.image(img_path, caption="Uploaded Image", use_column_width=True)
    
    # Load model
    model = load_model()
    
    # Generate caption
    with st.spinner("Generating caption..."):
        caption = generate_caption(model, img_path, vectorization)
    
    st.success(f"Generated Caption: {caption}")
    
    # Clean up temp file
    os.remove(img_path)
else:
    st.info("Please upload an image to generate a caption.")

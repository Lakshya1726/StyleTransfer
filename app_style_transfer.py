import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
from io import BytesIO


st.set_page_config(page_title="Image Style Transfer", layout="centered")



@st.cache_resource
def load_model():
    """
    Loads the pre-trained style transfer model from TensorFlow Hub.
    Using st.cache_resource ensures the model is loaded only once.
    """
    model_path = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
    model = hub.load(model_path)
    return model


def crop_and_resize(image, image_size=(256, 256)):
    """Crops and resizes an image to the target size."""
    image = tf.image.convert_image_dtype(image, tf.float32)
    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    long_dim = max(shape)
    scale = image_size[0] / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)
    image = tf.image.pad_to_bounding_box(image, 0, 0, image_size[0], image_size[1])
    return image


def run_style_transfer(model, content_image, style_image):
    """
    Runs the style transfer process.
    """
    content_image = tf.constant(np.array(content_image))
    style_image = tf.constant(np.array(style_image))

    content_image = crop_and_resize(content_image)
    style_image = crop_and_resize(style_image)

    content_image = tf.expand_dims(content_image, axis=0)
    style_image = tf.expand_dims(style_image, axis=0)


    stylized_image = model(tf.image.convert_image_dtype(content_image, tf.float32),
                           tf.image.convert_image_dtype(style_image, tf.float32))[0]

    stylized_image = tf.squeeze(stylized_image, axis=0)
    stylized_image = np.array(stylized_image) * 255
    stylized_image = Image.fromarray(stylized_image.astype('uint8'))

    return stylized_image


st.title("🎨 Image Style Transfer")
st.markdown(
    "Upload two images: a **content** image and a **style** image. The app will generate a new image that combines the content of the first with the artistic style of the second.")

model = load_model()


col1, col2 = st.columns(2)

with col1:
    content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
    if content_file:
        content_image = Image.open(content_file).convert("RGB")
        st.image(content_image, caption="Content Image", use_column_width=True)

with col2:
    style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])
    if style_file:
        style_image = Image.open(style_file).convert("RGB")
        st.image(style_image, caption="Style Image", use_column_width=True)


if st.button("Apply Style Transfer", use_container_width=True):
    if content_file and style_file:
        with st.spinner("Applying style transfer... This may take a moment."):
            try:
                stylized_image = run_style_transfer(model, content_image, style_image)
                st.subheader("Stylized Image")
                st.image(stylized_image, caption="Stylized Image", use_column_width=True)

                
                stylized_image_bytes = BytesIO()
                stylized_image.save(stylized_image_bytes, format='PNG')
                st.download_button(
                    label="Download Stylized Image",
                    data=stylized_image_bytes.getvalue(),
                    file_name="stylized_image.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload both a content and a style image.")

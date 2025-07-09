import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load the model (replace 'chest_xray_model.h5' with your actual file path)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('basic_cnn_model.h5')
    return model

model = load_model()

st.title("Chest X-Ray Pneumonia Detection")

uploaded_file = st.file_uploader("Choose a Chest X-Ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-Ray.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    size = (150, 150)  # Adjust to your model's input size
    image = ImageOps.fit(image, size, Image.LANCZOS)
    img_array = np.asarray(image)
    img_array = img_array.astype('float32') / 255.0  # Normalize
    img_array = np.expand_dims(img_array, 0)  # Create batch dimension

    # Make prediction
    print(f"Model input shape: {model.input_shape}")
    print(f"Image array shape: {img_array.shape}")
    predictions = model.predict(img_array)
    class_names = ['Normal', 'Pneumonia']  # Adjust to your model's class names
    score = tf.nn.softmax(predictions[0])

    st.write(
        "This X-Ray most likely shows {} with a {:.2f}% confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    st.write(f"Raw prediction values: {predictions}") #For debugging purposes.

else:
    st.info("Please upload a Chest X-Ray image for classification.")
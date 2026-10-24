import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Model load karein (Caching taaki baar baar load na ho)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('animal_detector_model.keras')
    return model

model = load_model()

# Classes ki list (jo training ke waqt thin)
class_labels = ['cats', 'elephants', 'giraffes', 'lions', 'pandas', 'tigers']

st.title("🐾 Animal Detection AI Model")
st.write("Apne animal ki image upload karein (Cats, Elephants, Giraffes, Lions, Pandas, Tigers). Agar koi human ya doosra object hoga toh model bata dega ke ye animal nahi hai.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Image display karein
    image_obj = Image.open(uploaded_file)
    st.image(image_obj, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess aur Predict
    img = image_obj.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    max_prob = np.max(predictions)
    predicted_class_idx = np.argmax(predictions)
    
    # Threshold check (60% confidence)
    THRESHOLD = 0.6
    
    st.write("---")
    if max_prob < THRESHOLD:
        st.error("❌ Yeh koi animal nahi hai (Not an Animal / Unknown object)")
    else:
        animal_name = class_labels[predicted_class_idx]
        st.success(f"✅ Detected Animal: **{animal_name.upper()}**")
        st.info(f"Confidence Score: {max_prob * 100:.2f}%")
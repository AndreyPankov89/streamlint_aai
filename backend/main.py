import io
import streamlit as st
import json
from PIL import Image
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

@st.cache(allow_output_mutation=True)
def load_model():
    return models.load_model('st_dogs.keras')


def preprocess_image(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение для распознавания')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def print_predictions(preds):
    top = sorted(zip(list(preds), list(classes)), reverse=True)[:3]
    for item in top:
        st.write(f'**{item[1]}**, {item[0]}')




model = load_model()
with open('classes.json', 'r') as f:
    classes = json.load(f)


st.title('классификация пород собак Streamlit')
img = load_image()
result = st.button('Распознать изображение')
if result:
    x = preprocess_image(img)
    preds = model.predict(x)
    st.write('**Результаты распознавания:**')
    print_predictions(preds[0])
import streamlit as st
import numpy as np
from tensorflow import keras
import keras.utils as image
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Upload a satellite image", page_icon=":satellite:")
st.title('Upload a satellite image')

@st.cache(allow_output_mutation=True)
def load_model():
    model = keras.models.load_model('model3.h5', compile=False)
    return model
    
@st.cache(allow_output_mutation=True)
def load_modelseg():
    model = keras.models.load_model('silo_segmentation_last_adagrad.h5', compile=False)
    return model
    
model = load_model()

model_seg = load_modelseg()

img_width, img_height = 256, 256

upload = st.file_uploader("Upload your file(s) here:", type=['png','jpg'], accept_multiple_files=True)

if upload is not None:

    for uploaded_file in upload:

        img = Image.open(uploaded_file)
        img = img.resize((img_width, img_height))

        c1, c2, c3 = st.columns((1, 1, 1))

        with c1:
            st.image(img)

        #out = open('temp.png', 'wb')
        #out.write(Image.open(upload))
        #out.close()

        with c2:

            with st.spinner('Wait for it...'):

                #img = image.load_img('temp.png', target_size = (img_width, img_height))
                img = image.img_to_array(img)
                img= img.astype('float32') / 255.
                img = np.expand_dims(img, axis = 0)
                prob = model.predict(img)

                st.write(f"Forecasted probability that the picture contains silo(s): {prob[0][0]*100:.0f}%")
                if prob[0][0]>=0.5:
                    st.success(f"Silos have been identified in this picture.", icon="✅")
                    with c3:
                        img = img * 255
                        prob_seg = model_seg.predict(img)
                        prob_seg = prob_seg[0].reshape(256,256) * 255
                        prob_seg = Image.fromarray(prob_seg)
                        prob_seg = prob_seg.convert("RGB")
                        st.image(prob_seg)
                else:
                    st.error(f"No silos have been identified in this picture.", icon="❌")
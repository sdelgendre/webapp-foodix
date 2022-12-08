import folium as fl
from streamlit_folium import st_folium
import streamlit as st
from owslib.wms import WebMapService

import numpy as np
from tensorflow import keras
import keras.utils as image
from PIL import Image

import os
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

import pydeck as pdk
from pydeck.types import String

st.set_page_config(layout="wide", page_title="Draw silos heatmap", page_icon=":ear_of_rice:")
st.title('Draw silos heatmap')

@st.cache(allow_output_mutation=True)
def load_model():
    model = keras.models.load_model('model3.h5', compile=False)
    return model

def get_pos(lat,lng):
    return lat,lng
    
model = load_model()

img_width, img_height = 256, 256

c1, c2, c3 = st.columns((1, 1, 2))

locations = pd.DataFrame()

wms = WebMapService("https://ortho.gis.iastate.edu/arcgis/services/ortho/naip_2021_nc/ImageServer/WMSServer?")
name = "naip_2021_nc"

wms.getOperationByName('GetMap').methods[0]["url"] = "https://ortho.gis.iastate.edu/arcgis/services/ortho/naip_2021_nc/ImageServer/WMSServer"

with c1:

    min_lat = st.text_input("Latitude min.", value="40.899")
    max_lat = st.text_input("Latitude max.", value="41.157")

    min_lng = st.text_input("Longitude min.", value="-94.470")
    max_lng = st.text_input("Longitude max.", value="-94.014")

    min_lat = round(float(min_lat), 3)
    max_lat = round(float(max_lat), 3)
    min_lng = round(float(min_lng), 3)
    max_lng = round(float(max_lng), 3)

with c2:

    df_in_temp = pd.DataFrame(columns = ["longitude", "latitude"])

    with st.spinner('Loading...'):
        for filename in os.listdir('temp/test'):
            f = os.path.join('temp/test', filename)
            # checking if it is a file
            if os.path.isfile(f) and filename.endswith('.png'):
                filename = filename.replace('.png', '')
                lng, lat = filename.split("_")
                
                #img = Image.open(f).convert('RGB')
                #img = image.img_to_array(img)
                #img= img.astype('float32') / 255.
                #img = np.expand_dims(img, axis = 0)
                #prob = model.predict(img)

                df_in_temp.loc[len(df_in_temp)] = [float(lng), float(lat)]
    
    ln_temp = len(df_in_temp.loc[(df_in_temp["longitude"] >= min_lng) & 
                    (df_in_temp["longitude"] <= max_lng) &
                    (df_in_temp["latitude"] >= min_lat) &
                    (df_in_temp["latitude"] <= max_lat)])

    st.write(str(ln_temp) + "/" + str(len(np.arange(min_lng, max_lng, 0.002))*len(np.arange(min_lat, max_lat, 0.002))) + " pictures already collected.")

    collect = st.button('Collect additional data')

    sub_an = st.button('Launch analysis over collected data')

if collect:
    with st.spinner('Wait for it...'):
        for i in np.arange(min_lng, max_lng, 0.002):
        #for i in np.arange(-94.470, -94.216, 0.002):
            for j in np.arange(min_lat, max_lat, 0.002):
            #for j in np.arange(40.899, 41.009, 0.002):

                i = round(i, 3)
                j = round(j, 3)

                if not os.path.exists('temp/test/'+str(i)+'_'+str(j)+'.png'):

                    response = wms.getmap(
                        layers=[
                            name,
                        ],
                        # Left, bottom, right, top
                        #bbox=(-93.746323, 41.326376, -93.735312, 41.334719),
                        bbox=(i-0.001, j-0.001, i+0.001, j+0.001),
                        format="image/png",
                        size=(256, 256),
                        srs="EPSG:4326",
                        transparent=False,
                    )

                    out = open('temp/test/'+str(i)+'_'+str(j)+'.png', 'wb')
                    out.write(response.read())
                    out.close()
    
if sub_an:

    df = pd.DataFrame(columns = ["longitude", "latitude", "label", "prob"])

    for filename in os.listdir('temp/test'):
        f = os.path.join('temp/test', filename)
        # checking if it is a file
        if os.path.isfile(f) and filename.endswith('.png'):
            filename = filename.replace('.png', '')
            lng, lat = filename.split("_")
            
            #img = Image.open(f).convert('RGB')
            #img = image.img_to_array(img)
            #img= img.astype('float32') / 255.
            #img = np.expand_dims(img, axis = 0)
            #prob = model.predict(img)

            df.loc[len(df)] = [float(lng), float(lat), False, 0]
    
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_set = test_generator = test_datagen.flow_from_directory(
        'temp/',
        target_size=(256,256),
        batch_size=32,
        class_mode=None,
        shuffle=False,
    )
    
    predictions = model.predict(test_set)

    df["prob"] = predictions.flatten()
    
    #st.write(df.loc[df["label"] == True])
    df_true = df.loc[df["prob"] >= 0.5]
    locations = df_true[['latitude', 'longitude']]

with c3:

    #if len(locations) > 0:
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=(40.899+40.999)/2,
            longitude=(-94.470+-94.370)/2,
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'HeatmapLayer',
                data=locations,
                get_position='[longitude, latitude]',
                opacity=0.9,
                aggregation=String('MEAN'),
            ),
            ],
    ))

    #locationlist = locations.values.tolist()
    #for point in range(0, len(locationlist)):
    #    fl.Marker(locationlist[point], popup=' ').add_to(m)

    #with c1:
    #    st_folium(m, height=500, width=700)

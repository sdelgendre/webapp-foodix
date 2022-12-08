import folium as fl 
import numpy as np
import pandas as pd
import streamlit as st
from haversine import haversine
from streamlit_folium import st_folium
from pages.utils import monitoring_dashboard


st.set_page_config(layout="wide", page_title="Monitor my silos", page_icon=":gear:")
st.title('Monitor my silos')

fillValues = pd.read_csv("data/fill_values_silos.csv")

silos = pd.DataFrame(
    {
        "name": ["Syracuse", "Creston", "Clarinda"],
        "lat": [40.6, 41.0, 40.7],
        "lon": [-96.0, -94.5, -95.0],
        "temperature": [8.5, 5.9, 9.0],
        "humidity": [0.45, 0.53, 0.58],
        "pressure": [994, 1007, 980],
        "total_capacity": [7.2, 9.5, 11],
        "remaining_capacity": [4.8, 4.2, 6.2]
    }
)

c1, c2 = st.columns((1, 1))

with c1:
    m = fl.Map(
        location=[(40.326376+41.579830)/2, (-96.746323-93.791328)/2],
        zoom_start=8
    )

    for index, row in silos.iterrows():
        fl.Marker(
            location=[row.loc["lat"], row.loc["lon"]],
            tooltip = "name: {} capacity: {}".format(row.loc["name"], row.loc["total_capacity"])
        ).add_to(m)
    
    map = st_folium(m, height=500, width=700)

with c2:
    try:
        coords = map['last_clicked']['lat'], map['last_clicked']['lng']
    except: 
        coords = (41.579830, -93.791328)

    silos["distance_to_last_click"] = silos.apply(lambda row: haversine((row["lat"], row["lon"]), coords), axis=1)
    closest_silo = silos["distance_to_last_click"].idxmin()

    c2.pyplot(monitoring_dashboard.plot_monitoring_dashboard(silos, closest_silo, fillValues))

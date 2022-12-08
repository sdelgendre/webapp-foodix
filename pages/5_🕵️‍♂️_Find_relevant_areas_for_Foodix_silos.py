import numpy as np
import pandas as pd
import folium
import branca
from folium import plugins
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import geojsoncontour
import scipy as sp
import scipy.ndimage
import streamlit as st
from streamlit_folium import st_folium
from streamlit_folium import folium_static
import asyncio
import time

st.set_page_config(layout="wide", page_title="Rank areas of interest", page_icon=":sleuth_or_spy:")
st.title("Finding areas of interest for Foodix silos")
st.subheader('Based on climatic conditions in Iowa')

c1, c2, c3 = st.columns((1, 2, 1))

with c1:
    st.write("")

with c3:
    st.write("")

with c2:

    st.markdown("Potential is a score based on Temperature and Humidity level")

    # Setup
    score_mean = 50
    score_std = 5
    debug = False

    # Setup colormap
    colors = ['#D7191C', '#FDAE61', '#FFFFBF', '#ABDDA4', '#2B83BA']
    vmin   = score_mean - 2 * score_std
    vmax   = score_mean + 2 * score_std
    levels = len(colors)
    cm     = branca.colormap.LinearColormap(colors, vmin=vmin, vmax=vmax).to_step(levels)

    # Create a dataframe with fake data
    rs = 1
    xmin, ymin, xmax, ymax = (-96.639704, 40.375501, -90.140061, 43.501196)
    longitude = [i for j in np.random.uniform(xmin, xmax, size=(100, 1)) for i in j]
    latitude  = [i for j in np.random.uniform(ymin, ymax, size=(100, 1)) for i in j]
    temperature = [i for j in np.random.normal(50, 5, size=(100, 1)) for i in j]
    df = pd.DataFrame(
        list(zip(latitude, longitude, temperature)),
        columns=['latitude', 'longitude', 'temperature']
    )

    # The original data
    x_orig = np.asarray(df.longitude.tolist())
    y_orig = np.asarray(df.latitude.tolist())
    z_orig = np.asarray(df.temperature.tolist())

    # Make a grid
    x_arr          = np.linspace(np.min(x_orig), np.max(x_orig), 500)
    y_arr          = np.linspace(np.min(y_orig), np.max(y_orig), 500)
    x_mesh, y_mesh = np.meshgrid(x_arr, y_arr)

    # Grid the values
    z_mesh = griddata((x_orig, y_orig), z_orig, (x_mesh, y_mesh), method='linear')

    # Gaussian filter the grid to make it smoother
    sigma = [5, 5]
    z_mesh = sp.ndimage.gaussian_filter(z_mesh, sigma, mode='constant')


    table = st.empty()

    while True:

        # Create the contour
        fig, ax = plt.subplots()
        contourf = ax.contourf(x_mesh, y_mesh, z_mesh, levels, alpha=0.5, colors=colors, linestyles='None', vmin=vmin, vmax=vmax)

        # Convert matplotlib contourf to geojson
        geojson = geojsoncontour.contourf_to_geojson(
                                                        contourf=contourf,
                                                        min_angle_deg=3.0,
                                                        ndigits=5,
                                                        stroke_width=1,
                                                        fill_opacity=0.5)

        # Set up the folium plot
        geomap = folium.Map([df.latitude.mean(), df.longitude.mean()], zoom_start=7, tiles="cartodbpositron")

        # Plot the contour plot on folium
        folium.GeoJson(
        geojson,
        style_function=lambda x: {
                                    'color': x['properties']['stroke'],
                                    'weight':x['properties']['stroke-width'],
                                    'fillColor': x['properties']['fill'],
                                    'opacity':   0.8,
                                    }).add_to(geomap)

        # Add the colormap to the folium map
        cm.caption = 'Potential for implantation'
        geomap.add_child(cm)

        map = st_folium(geomap, height=500, width=700)

        st.markdown("This map displays areas where Foodix should create (or replace) silos 🟦") 
        st.markdown("  and where Foodix should not invest 🟥")

        # update every 5 mins
        time.sleep(600)  
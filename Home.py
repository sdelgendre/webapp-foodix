import streamlit as st

st.sidebar.title("About")
st.sidebar.info(
    """
    GitHub repository: <https://github.com/Windu12345/Quantum-Black-Challenge>
    """
)

st.markdown(
    """
    # 👋 Welcome to the Iowa silos monitoring app!

    Navigate through the **sidebar tabs** to enjoy some powerful and efficient tools developed by our team.

    ## Want to know more about our features?
    - [Browse silos in Iowa](Browse_silos_in_Iowa) to analyze every corner of Iowa's territory and experiment our latest computer vision models,
    - [Upload your satellite data](Analyze_satellite_data) to find out silos on your own land imagery,
    - [Map all the silos in Iowa](Draw_silos_heatmap) through our gridding tool,
    - [Monitor your silos](Monitor_my_silos) and reach out real time hygrometric parameters in your silos,
    - [Find relevant areas for your silos](Find_relevant_areas_for_Foodix_silos) based on climatic conditions in Iowa.

    """
)
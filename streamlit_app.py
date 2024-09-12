import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.title("🪐 NASA ML Application 🪐")

st.info(
    """
    **Discover Space's Close Encounters!**\\
    Near Earth Objects (N.E.O.s) are celestial bodies that come close to Earth. While most are harmless, some are tagged by NASA as *hazardous*. 
    Explore their history, from the first recorded sighting in 1910 to today.
    """
)

df = pd.read_csv("https://raw.githubusercontent.com/mateuszwalo/NASA_app/master/Nasa_clean_v2.csv")
X=df.drop("is_hazardous",axis=1)
y=df["is_hazardous"]

with st.expander("🔍 View NASA's Data 🔍"):
    st.write("**Explore NASA's extensive database, documenting every recorded N.E.O. from 1910 to 2024.**")
    st.dataframe(df)

with st.expander("📈 Data Visualization 📈"):
    selected_column = st.selectbox("Choose column", df.columns)
    if selected_column:
        fig = px.histogram(df, x=selected_column, title=f'Histogram for: {selected_column}')
        st.plotly_chart(fig)


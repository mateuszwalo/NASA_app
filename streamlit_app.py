import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🪐 NASA ML Application 🪐")

st.info(
    """
    **Explore the Wonders and Dangers of Space!**\\
    The universe is full of celestial bodies, but some come perilously close to Earth. These are known as **N.E.O.s - Near Earth Objects**. 
    While most of these space wanderers harmlessly pass by, a few are classified as *hazardous* by NASA, posing potential threats to our planet.  
    Dive into the data and uncover the history of these fascinating objects, from their first recorded observation in 1910 to the present day.
    """
)


df = pd.read_csv("https://raw.githubusercontent.com/mateuszwalo/NASA_app/master/Nasa_clean_v2.csv")

with st.expander("🔍 View NASA's Data Records"):
    st.write("**Explore NASA's extensive database, documenting every recorded N.E.O. from 1910 to 2024.**")
    st.dataframe(df)

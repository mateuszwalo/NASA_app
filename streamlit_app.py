import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🪐 NASA ML Application 🪐")
st.write("Author: Mateusz Walo")
st.info("There are many dangerous bodies in space, one of them is N.E.O. - *Nearest Earth Objects*. Some such bodies really pose a danger to the planet Earth, NASA classifies them as *is_hazardous*.")
df=pd.read_csv("https://raw.githubusercontent.com/mateuszwalo/NASA_app/master/Nasa_clean_v2.csv")

with st.expander("Data"):
  st.write("**This dataset contains ALL NASA observations of similar objects from 1910 to 2024**")
  df

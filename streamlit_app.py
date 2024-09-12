import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🪐 NASA ML Application 🪐")
st.info('This application will allow you to do EDA and predict the threat to Earth thanks to machine learning models')

df=pd.read_csv("https://raw.githubusercontent.com/mateuszwalo/NASA_app/master/Nasa_clean.csv")
df

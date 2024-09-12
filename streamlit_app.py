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

plots_type = ["Histogram", "Box Plot", "Scatter Plot","Bar Plot"]  

with st.expander("📈 Data Visualization 📈"):
    selected_plot_type = st.selectbox("Choose type of plot", plots_type)
    if selected_plot_type == "Histogram":
        num_bins = st.slider("Number of bins", min_value=10, max_value=100, value=50)
        selected_column = st.selectbox("Choose column", X.columns)
        if selected_column:
            fig = px.histogram(
                df, 
                x=selected_column, 
                nbins=num_bins,
                title=f'Histogram for: {selected_column}'
            )
            st.plotly_chart(fig)
    elif selected_plot_type == "Box Plot":
        selected_column = st.selectbox("Choose column for Box Plot", X.columns)
        if selected_column:
            fig = px.box(
                df, 
                y=selected_column,
                title=f'Box Plot for: {selected_column}'
            )
            st.plotly_chart(fig)
    
    elif selected_plot_type == "Scatter Plot":
        x_column = st.selectbox("Choose x-axis column for Scatter Plot", X.columns)
        y_column = st.selectbox("Choose y-axis column for Scatter Plot", X.columns)
        
        if x_column and y_column:
            fig = px.scatter(
                df, 
                x=x_column, 
                y=y_column,
                title=f'Scatter Plot: {x_column} vs {y_column}'
            )
            st.plotly_chart(fig)
            
    elif selected_plot_type == "Bar Plot":
        column=st.selectbox("Choose column for Bar Plot", y.columns)
        fig=px.bar(df,
                  x=column,
                  title=f"Bar plot for {column}")
        st.plotly_chart(fig)


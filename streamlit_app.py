import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
from sklearn.model_selection import train_test_split

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

with st.expander("📡 Features Descriptions 📡"):
    st.write("**...**")
 
with st.expander("📈 Data Visualization 📈"):
    plots_type = ["Histogram", "Box Plot", "Scatter Plot"] 
    selected_plot_type = st.selectbox("Choose type of plot", plots_type)
    if selected_plot_type == "Histogram":
        num_bins = st.slider("Number of bins", min_value=10, max_value=100, value=50)
        selected_column = st.selectbox("Choose column", df.columns)
        if selected_column:
            fig = px.histogram(
                df, 
                x=selected_column, 
                nbins=num_bins,
                title=f'Histogram for: {selected_column}',
            )
            fig.update_traces(marker_color='purple')
            st.plotly_chart(fig)
    elif selected_plot_type == "Box Plot":
        selected_column = st.selectbox("Choose column for Box Plot", df.columns)
        if selected_column:
            fig = px.box(
                df, 
                y=selected_column,
                title=f'Box Plot for: {selected_column}'
            )
            fig.update_traces(marker_color='purple')
            st.plotly_chart(fig)
    
    elif selected_plot_type == "Scatter Plot":
        x_column = st.selectbox("Choose x-axis column for Scatter Plot", df.columns)
        y_column = st.selectbox("Choose y-axis column for Scatter Plot", df.columns)
        
        if x_column and y_column:
            fig = px.scatter(
                df, 
                x=x_column, 
                y=y_column,
                title=f'Scatter Plot: {x_column} vs {y_column}'
            )
            fig.update_traces(marker_color='purple')
            st.plotly_chart(fig)

with st.expander("🎯 Statistics 🎯"):
    stat_options = ["Correlation Matrix", "Descriptive Statistics"]
    selected_stat = st.selectbox("Choose statistical analysis", stat_options)

    if selected_stat == "Correlation Matrix":
        cm=df.corr()
        st.subheader("Correlation Matrix")
        fig = px.imshow(cm.round(2), color_continuous_scale='YlOrRd',text_auto=True)
        st.plotly_chart(fig)

    elif selected_stat == "Descriptive Statistics":
        st.subheader("Descriptive Statistics")
        column = st.selectbox("Choose column for Descriptive Statistics", df.columns)
        if column:
            st.write(df[column].describe())
with st.expander("⚙️ Model training ⚙️"):
    st.info("In this part you can train your custom model to predict NEO`s ")
    t_size=st.slider("Test size (recomended = 0.2)", min_value=0.1, max_value=0.9, value=0.2)
    X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=t_size, stratify=y)
    st.write(f"Train X size = {X_train.shape}")
    st.write(f"Train y size = {y_train.shape}")
    st.write(f"Test X size = {X_test.shape}")
    st.write(f"Train y size = {y_test.shape}")
    
    


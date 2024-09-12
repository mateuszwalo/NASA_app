import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats

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
    stat_options = ["Correlation Matrix", "Statistical Tests", "Descriptive Statistics", "Probability Distributions"]
    selected_stat = st.selectbox("Choose statistical analysis", stat_options)

    if selected_stat == "Correlation Matrix":
        st.subheader("Correlation Matrix")
        correlation_matrix = df.corr()
        st.write(correlation_matrix)
        fig = px.imshow(correlation_matrix, color_continuous_scale='Viridis', title="Correlation Matrix")
        st.plotly_chart(fig)

    elif selected_stat == "Statistical Tests":
        st.subheader("Statistical Tests")
        test_type = st.selectbox("Choose test", ["Normality Test", "T-Test", "ANOVA", "Chi-Square Test"])
        
        if test_type == "Normality Test":
            column = st.selectbox("Choose column for Normality Test", df.columns)
            if column:
                stat, p_value = stats.shapiro(df[column].dropna())
                st.write(f"Shapiro-Wilk Test Statistic: {stat}")
                st.write(f"P-Value: {p_value}")
        
        elif test_type == "T-Test":
            column1 = st.selectbox("Choose first column for T-Test", df.columns)
            column2 = st.selectbox("Choose second column for T-Test", df.columns)
            if column1 and column2:
                t_stat, p_value = stats.ttest_ind(df[column1].dropna(), df[column2].dropna())
                st.write(f"T-Statistic: {t_stat}")
                st.write(f"P-Value: {p_value}")
        
        elif test_type == "ANOVA":
            column = st.selectbox("Choose column for ANOVA", df.columns)
            group = st.selectbox("Choose group column", df.columns)
            if column and group:
                groups = df.groupby(group)[column].apply(list)
                f_stat, p_value = stats.f_oneway(*groups)
                st.write(f"ANOVA F-Statistic: {f_stat}")
                st.write(f"P-Value: {p_value}")
        
        elif test_type == "Chi-Square Test":
            observed = st.text_input("Enter observed frequencies (comma-separated)", "")
            expected = st.text_input("Enter expected frequencies (comma-separated)", "")
            if observed and expected:
                observed = np.array(list(map(int, observed.split(','))))
                expected = np.array(list(map(int, expected.split(','))))
                chi2_stat, p_value = stats.chisquare(observed, expected)
                st.write(f"Chi-Square Statistic: {chi2_stat}")
                st.write(f"P-Value: {p_value}")

    elif selected_stat == "Descriptive Statistics":
        st.subheader("Descriptive Statistics")
        column = st.selectbox("Choose column for Descriptive Statistics", df.columns)
        if column:
            st.write(df[column].describe())

    elif selected_stat == "Probability Distributions":
        st.subheader("Probability Distributions")
        column = st.selectbox("Choose column for Probability Distribution", df.columns)
        if column:
            fig = px.histogram(df, x=column, marginal="box", title=f"Distribution of {column}")
            st.plotly_chart(fig)

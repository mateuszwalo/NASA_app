import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, cohen_kappa_score,roc_curve, roc_auc_score

def model_evaluation(classifier, x_test, y_test):
    cm = confusion_matrix(y_test, classifier.predict(x_test))
    names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    counts = [value for value in cm.flatten()]
    percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names, counts, percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Purples', cbar=False, annot_kws={"size": 14})
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    st.pyplot(plt)

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
    st.info("In this section, you can train your custom model with to predict NEOs (Near-Earth Objects).\\Due to the imbalanced target, SMOTE was used to upsample the minority class, and the data has been standardized for better model performance.")
    
    X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2, stratify=y)
    smote=SMOTE(k_neighbors=3, random_state=10)
    X_train, y_train=smote.fit_resample(X_train,y_train)
    transformer = StandardScaler()
    transformer2=transformer.fit(X_train)
    X_train = transformer2.transform(X_train)
    X_test = transformer2.transform(X_test)
    models=["Decision Tree","Random Forest","XGB Classifer","Logistic Regression"]
    
    selected_model=st.selectbox("Choose model ",models)
    
    if selected_model == "Logistic Regression":
        st.subheader("Logistic Regression - Model Configuration")
        c_value = st.slider("C (Inverse of regularization strength)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        max_iter = st.slider("Maximum Iterations", min_value=50, max_value=500, value=100, step=10)
        solver = st.selectbox("Solver", ["lbfgs", "liblinear", "sag", "saga"])
        lr = LogisticRegression(C=c_value, max_iter=max_iter, solver=solver, random_state=10)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        Accuracy_lr = accuracy_score(y_test, y_pred_lr)
        F1_lr = f1_score(y_test, y_pred_lr)
        Kappa_lr = cohen_kappa_score(y_test, y_pred_lr)
    
        st.subheader("Logistic Regression - Evaluation Metrics")
        st.write(f"**Accuracy in Logistic Regression =** {Accuracy_lr}")
        st.write(f"**F1 in Logistic Regression =** {F1_lr}")
        st.write(f"**Kappa in Logistic Regression =** {Kappa_lr}")
        model_evaluation(lr, X_test, y_test)
        st.subheader("ROC Curve and AUC")
        y_pred_proba_lr = lr.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_lr)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='purple', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        st.pyplot(plt)


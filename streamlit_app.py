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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, cohen_kappa_score,roc_curve, roc_auc_score, auc
import pickle

pickle_in_1=open("tree.pkl","rb")
best_tree=pickle.load(pickle_in_1)

def predict_own_neo(absolute_magnitude, estimated_diameter_min, estimated_diameter_max, relative_velocity, miss_distance):
    try:
        prediction = best_tree.predict([[absolute_magnitude, estimated_diameter_min, estimated_diameter_max, relative_velocity, miss_distance]])
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None
    

def model_evaluation(classifier, x_test, y_test):
    from sklearn.linear_model import LogisticRegression
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

def plot_roc_curve(y_test, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
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
    
with st.sidebar:
    st.header("About")
    st.sidebar.info(
        """
        **Author:** *Mateusz Walo*  
        **LinkedIn:** [My LinkedIn profile :)](https://www.linkedin.com/in/mateuszwalo/)  
        **Data:** [Nearest Earth Objects (1910-2024)](https://www.kaggle.com/datasets/ivansher/nasa-nearest-earth-objects-1910-2024)
        """
    )
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

with st.expander("📡 Columns Descriptions 📡"):
    st.info("**absolute_magnitude** - describes intrinsic luminosity")
    st.info("**estimated_diameter_min** -minimum Estimated Diameter in Kilometres")
    st.info("**estimated_diameter_max** - maximum Estimated Diameter in Kilometres")
    st.info("**relative_velocity** - velocity Relative to Earth in Kmph")
    st.info("**miss_distance** - distance in Kilometres missed")
    st.info("**is_hazardous** - Boolean feature that shows whether asteroid is harmful or no")
    
 
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
        selected_column = st.selectbox("Choose column for Box Plot", X.columns)
        if selected_column:
            fig = px.box(
                df, 
                y=selected_column,
                title=f'Box Plot for: {selected_column}'
            )
            fig.update_traces(marker_color='purple')
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
    st.info(
        "In this section, you can train your custom model to predict NEOs (Near-Earth Objects). "
        "Due to the imbalanced target, SMOTE was used to upsample the minority class, and the data has been standardized for better model performance."
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    smote = SMOTE(k_neighbors=3, random_state=10)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    models = ["Decision Tree", "Random Forest", "XGB Classifier", "Logistic Regression"]
    selected_model = st.selectbox("Choose model", models)
    
    if selected_model == "Decision Tree":
        from sklearn.tree import DecisionTreeClassifier
        st.subheader("Decision Tree - Model Configuration")
        max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=3, step=1)
        min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, value=2, step=1)
        min_samples_leaf = st.slider("Min Samples Leaf", min_value=1, max_value=20, value=1, step=1)
        
        dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=42)
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)
        
        Accuracy_dt = accuracy_score(y_test, y_pred_dt)
        F1_dt = f1_score(y_test, y_pred_dt)
        Kappa_dt = cohen_kappa_score(y_test, y_pred_dt)
        
        st.subheader("Decision Tree - Evaluation Metrics")
        st.write(f"**Accuracy in Decision Tree =** {Accuracy_dt}")
        st.write(f"**F1 in Decision Tree =** {F1_dt}")
        st.write(f"**Kappa in Decision Tree =** {Kappa_dt}")
        
        model_evaluation(dt, X_test, y_test)
        
        y_pred_proba_dt = dt.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test, y_pred_proba_dt)
    
    elif selected_model == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        st.subheader("Random Forest - Model Configuration")
        n_estimators = st.slider("Number of Estimators", min_value=10, max_value=300, value=100, step=10)
        max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=3, step=1)
        min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, value=2, step=1)
        
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        
        Accuracy_rf = accuracy_score(y_test, y_pred_rf)
        F1_rf = f1_score(y_test, y_pred_rf)
        Kappa_rf = cohen_kappa_score(y_test, y_pred_rf)
        
        st.subheader("Random Forest - Evaluation Metrics")
        st.write(f"**Accuracy in Random Forest =** {Accuracy_rf}")
        st.write(f"**F1 in Random Forest =** {F1_rf}")
        st.write(f"**Kappa in Random Forest =** {Kappa_rf}")
        
        model_evaluation(rf, X_test, y_test)
        
        y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test, y_pred_proba_rf)
    
    elif selected_model == "XGB Classifier":
        from xgboost import XGBClassifier
        st.subheader("XGB Classifier - Model Configuration")
        n_estimators = st.slider("Number of Estimators", min_value=10, max_value=300, value=100, step=10)
        max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=3, step=1)
        learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
        
        xgb = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        
        Accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
        F1_xgb = f1_score(y_test, y_pred_xgb)
        Kappa_xgb = cohen_kappa_score(y_test, y_pred_xgb)
        
        st.subheader("XGB Classifier - Evaluation Metrics")
        st.write(f"**Accuracy in XGB Classifier =** {Accuracy_xgb}")
        st.write(f"**F1 in XGB Classifier =** {F1_xgb}")
        st.write(f"**Kappa in XGB Classifier =** {Kappa_xgb}")
        
        model_evaluation(xgb, X_test, y_test)
        
        y_pred_proba_xgb = xgb.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test, y_pred_proba_xgb)
    
    elif selected_model == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression
        st.subheader("Logistic Regression - Model Configuration")
        c_value = st.slider("C (Inverse of regularization strength)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        max_iter_ = st.slider("Maximum Iterations", min_value=50, max_value=500, value=100, step=10)
        solver_ = st.selectbox("Solver", ["lbfgs", "liblinear", "sag", "saga"])
        
        try:
            lr = LogisticRegression(C=c_value, max_iter=max_iter_, solver=solver_, random_state=42)
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
            
            y_pred_proba_lr = lr.predict_proba(X_test)[:, 1]
            plot_roc_curve(y_test, y_pred_proba_lr)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

with st.expander("🎰 Predict your own NEO`s 🎰"):
    st.info("In this section you can check your own NEOs if they are a threat to the Earth")
    st.info("*Prediction with the best decision tree estimator*")
    absolute_magnitude = st.slider("Absolute magnitude", min_value=15, max_value=35, value=19, step=1)
    estimated_diameter_min = st.slider("Estimated diameter min", min_value=0.001, max_value=0.35, value=0.0015, step=0.0001)
    estimated_diameter_max = st.slider("Estimated diameter max", min_value=0.0025, max_value=0.8, value=0.1, step=0.0001)
    relative_velocity = st.slider("Relative velocity", min_value=200, max_value=120000, value=1000, step=10)
    miss_distance = st.slider("Miss distance", min_value=6000, max_value=75000000, value=10000, step=500)
    if st.button("**Predict**"):
        result = predict_own_neo(absolute_magnitude, estimated_diameter_min, estimated_diameter_max, relative_velocity, miss_distance)
        st.write(f'The output is: {result}')
        if result == 1:
            st.write("🚨 Warning! NEO is a threat to Earth! 🚨")
            st.image("https://media.giphy.com/media/LwIyvaNcnzsD6/giphy.gif", use_column_width=True)
        else: 
            st.write("😊 Relax! The Earth is safe!")
            st.image("https://media.giphy.com/media/3oEjI6SIIHBdRxXI40/giphy.gif", use_column_width=True)

    
    
    
            




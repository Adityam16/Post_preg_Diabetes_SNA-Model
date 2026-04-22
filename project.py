# =====================================
# Advanced Diabetes Prediction App (Streamlit)
# Using PIMA Dataset + Plotly 3D Visualization
# =====================================

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# -----------------------------
# Load PIMA Dataset
# -----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv("/Users/adityam/Downloads/diabetes.csv")
    return df

# -----------------------------
# Prepare Data
# -----------------------------
df = load_data()
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Diabetes Predictor", layout="wide")

st.title("🩺 Advanced Diabetes Prediction (Post Pregnancy)")
st.write(f"Model Accuracy: {accuracy*100:.2f}%")

# Sidebar Inputs
st.sidebar.header("Patient Input")

preg = st.sidebar.slider("Pregnancies", 0, 20, 1)
glucose = st.sidebar.slider("Glucose", 0, 200, 120)
bp = st.sidebar.slider("Blood Pressure", 0, 140, 70)
skin = st.sidebar.slider("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("Insulin", 0, 900, 80)
bmi = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.sidebar.slider("Age", 18, 90, 30)

input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
input_scaled = scaler.transform(input_data)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]

    if pred == 1:
        st.error(f"🔴 High Risk of Diabetes ({prob[1]*100:.2f}%)")
    else:
        st.success(f"🟢 Low Risk of Diabetes ({prob[0]*100:.2f}%)")

    # -----------------------------
    # 3D Visualization (Plotly)
    # -----------------------------
    st.subheader("🔍 Interactive 3D Visualization")

    viz_df = df.copy()

    fig = px.scatter_3d(
        viz_df,
        x="Glucose",
        y="BMI",
        z="Age",
        color="Outcome",
        title="3D Distribution of Patients",
    )

    # Add user point
    fig.add_scatter3d(
        x=[glucose],
        y=[bmi],
        z=[age],
        mode="markers",
        marker=dict(size=8),
        name="Your Input"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("Rotate, zoom, and explore the feature space interactively.")

# -----------------------------
# Extra: Dataset Preview
# -----------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("Feature Importance")
importances = model.feature_importances_
feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
feat_df = feat_df.sort_values(by="Importance", ascending=False)

st.bar_chart(feat_df.set_index("Feature"))


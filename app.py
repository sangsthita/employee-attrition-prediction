# =========================
# Streamlit Dashboard for Employee Attrition Prediction
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder  

# ----------------------------
# Load model and data
# ----------------------------
MODEL_PATH = "/Users/sangsthitapanda/Desktop/L&T PROJECT/models/attrition_pipeline.pkl"
DATA_PATH = "/Users/sangsthitapanda/Desktop/L&T PROJECT/archive/employee_attrition_with_sentiment.csv"

st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")

# Load model
model = joblib.load(MODEL_PATH)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Drop text-only columns that break sklearn
drop_cols = ["sentiment_label", "feedback_text"]

# -----------------------------
# Identify categorical/string columns
# -----------------------------
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Exclude target column if present
target_col = "Attrition"
if target_col in categorical_cols:
    categorical_cols.remove(target_col)

# -----------------------------
# Encode categorical columns
# -----------------------------
le_dict = {}  # Store encoders to use later if needed
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le


# Define features and target
X = df.drop(columns=drop_cols + ["attrition"], errors="ignore")
y = df["attrition"]


# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:", 
    ["📊 EDA", "📈 Analytics", "🤖 Model Performance", "🧑‍💼 Prediction"]
)

# ----------------------------
# 📊 EDA Page
# ----------------------------
if page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Dataset Overview")
    st.write(df.head())

    st.subheader("Attrition Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="attrition", palette="Set2", ax=ax)
    st.pyplot(fig)

    st.subheader("Numerical Feature Summary")
    st.write(df.describe())

    st.subheader("Categorical Feature Distribution")
    cat_features = df.select_dtypes(include="object").columns.tolist()
    feature = st.selectbox("Select a categorical feature:", cat_features)
    if feature:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=feature, hue="attrition", palette="coolwarm", ax=ax)
        plt.xticks(rotation=30)
        st.pyplot(fig)

# ----------------------------
# 📈 Analytics Page
# ----------------------------
elif page == "📈 Analytics":
    st.title("📈 Attrition Analytics")

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = df.select_dtypes(include=np.number).corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Interactive Feature vs Attrition
    st.subheader("📊 Explore Feature vs Attrition")
    feature = st.selectbox(
        "Choose a feature to analyze:",
        [col for col in df.columns if col not in ["attrition"]]
    )

    if df[feature].dtype == "object":
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=feature, hue="attrition", palette="coolwarm", ax=ax)
        plt.xticks(rotation=30)
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="attrition", y=feature, palette="coolwarm", ax=ax)
        st.pyplot(fig)

    # Sentiment vs Attrition (special view)
    if "sentiment_score" in df.columns:
        st.subheader("💬 Attrition vs Sentiment Score")
        fig, ax = plt.subplots()
        sns.kdeplot(
            data=df, x="sentiment_score", hue="attrition",
            fill=True, common_norm=False, palette="coolwarm", ax=ax
        )
        st.pyplot(fig)

    # Feature Importances
    st.subheader("🌟 Top Feature Importances (Model Driven)")
    try:
        if "classifier" in model.named_steps:
            feature_names = model.named_steps["preprocessor"].get_feature_names_out()
            importances = model.named_steps["classifier"].feature_importances_

            feat_imp = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(10)

            fig, ax = plt.subplots()
            sns.barplot(data=feat_imp, x="Importance", y="Feature", palette="Blues_r", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("This model does not support feature importance.")
    except Exception as e:
        st.warning(f"Feature importance not available: {e}")

# ----------------------------
# 🤖 Model Performance Page
# ----------------------------
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")

    st.subheader("Classification Report")
    X = df.drop(columns=["attrition"])
    y = df["attrition"]

    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Stay", "Leave"], yticklabels=["Stay", "Leave"], ax=ax)
    st.pyplot(fig)

# ----------------------------
# 🧑‍💼 Prediction Page
# ----------------------------
elif page == "🧑‍💼 Prediction":
    st.title("🧑‍💼 Employee Attrition Prediction")

    st.subheader("Enter Employee Details")
    input_data = {}
    for col in df.drop(columns=["attrition"]).columns:
        if df[col].dtype == "object":
            input_data[col] = st.selectbox(col, df[col].unique())
        else:
            input_data[col] = st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))

    if st.button("Predict Attrition"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        st.success(f"Prediction: {'Leave' if prediction == 1 else 'Stay'}")
        st.info(f"Probability of Leaving: {proba:.2f}")

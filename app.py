import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Streamlit app
st.title("Crocodile Conservation Status Prediction")

# File uploader
uploaded_file = st.file_uploader("Upload your crocodile dataset (CSV)", type="csv")

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)

    # Preprocessing: Drop unnecessary columns
    columns_to_drop = ["Observation ID", "Scientific Name", "Date of Observation", "Observer Name", "Notes"]
    df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)

    # Rename columns (fixed typo: "Common Nane" to "Common Name")
    df.rename(columns={
        "Common Name": "Name",  # Corrected from "Common Nane"
        "Observed Length (m)": "Length",
        "Observed Weight (kg)": "Weight",
        "Country/Region": "Region",
        "Habitat Type": "Habitat"
    }, inplace=True)


    # Define expected columns
    expected_columns = ["Name", "Length", "Weight", "Sex", "Age Class", "Family", "Genus", "Region", "Habitat", "Conservation Status"]

    # Check if all expected columns are present
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        st.error(f"The following required columns are missing in the dataset: {missing_columns}")
        st.stop()

    # Remove outliers
    def remove_outliers_iqr(df, col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return df[(df[col] >= lower) & (df[col] <= upper)]

    st.write("Before removing outliers:", df.shape)
    df = remove_outliers_iqr(df, "Length")
    df = remove_outliers_iqr(df, "Weight")
    st.write("After removing outliers:", df.shape)

    # Split features and target
    y = df["Conservation Status"]
    X = df.drop("Conservation Status", axis=1)

    # Define columns for preprocessing
    numeric_cols = ["Length", "Weight"]
    labelencode_cols = ["Sex", "Age Class"]
    onehot_cols = ["Name", "Family", "Genus", "Region", "Habitat"]

    # Validate preprocessing columns
    missing_preprocess_cols = []
    for col in numeric_cols + labelencode_cols + onehot_cols:
        if col not in X.columns:
            missing_preprocess_cols.append(col)
    if missing_preprocess_cols:
        st.error(f"The following columns required for preprocessing are missing: {missing_preprocess_cols}")
        st.stop()

    # Pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    label_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder())
    ])

    onehot_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_cols),
            ("label", label_pipeline, labelencode_cols),
            ("onehot", onehot_pipeline, onehot_cols)
        ],
        remainder="drop"
    )

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Transform data
    try:
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
    except Exception as e:
        st.error(f"Error during preprocessing: {str(e)}")
        st.stop()

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    # Train and evaluate models
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        st.write(f"\n### {name}")
        st.write(f"**Accuracy**: {acc:.4f}")
        st.write("**Classification Report**:")
        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        st.table(pd.DataFrame(report).transpose())

    # Visualize model performance
    st.write("### Model Comparison")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(results.keys(), results.values(), color="skyblue")
    ax.set_title("Model Comparison on Conservation Status Prediction")
    ax.set_ylabel("Accuracy")
    ax.set_xticklabels(results.keys(), rotation=20)
    st.pyplot(fig)

else:
    st.write("Please upload a CSV file to proceed.")
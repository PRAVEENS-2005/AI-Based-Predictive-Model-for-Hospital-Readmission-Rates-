# =========================
# FULL MODEL TRAINING CODE
# Hospital Readmission Prediction
# Trains Logistic Regression + Random Forest
# Saves deployable pipelines: lr_model.pkl and rf_model.pkl
# =========================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

# -------------------------
# 1) LOAD DATA
# -------------------------
DATA_PATH = "/home/praveen/ai/hospital_readmissions.csv"   # <-- change this file name if needed
TARGET_COL = "readmitted"               # <-- target column name in your dataset

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# -------------------------
# 2) BASIC CLEANING
# -------------------------
df = df.drop_duplicates()

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found. Available columns: {df.columns.tolist()}")

# Optional: if target is strings like "YES"/"NO", convert to 1/0
# If your target is already 0/1, this won't change anything meaningful.
if df[TARGET_COL].dtype == "object":
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().str.lower()
    df[TARGET_COL] = df[TARGET_COL].map({"yes": 1, "no": 0, "1": 1, "0": 0})
    if df[TARGET_COL].isna().any():
        raise ValueError("Target column has values not in {yes,no,0,1}. Please map them manually.")

# Split features/label
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(int)

# -------------------------
# 3) IDENTIFY COLUMN TYPES
# -------------------------
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=["number", "bool", "int64", "float64"]).columns.tolist()

print("Numeric columns:", len(num_cols))
print("Categorical columns:", len(cat_cols))

# -------------------------
# 4) PREPROCESSING PIPELINES
# -------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ],
    remainder="drop"
)

# -------------------------
# 5) TRAIN-TEST SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y if y.nunique() == 2 else None
)

print("Train:", X_train.shape, "Test:", X_test.shape)

# -------------------------
# 6) MODELS
# -------------------------
lr_clf = LogisticRegression(max_iter=1000)
rf_clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Full pipelines (preprocess + model)
lr_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", lr_clf)
])

rf_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", rf_clf)
])

# -------------------------
# 7) TRAIN
# -------------------------
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# -------------------------
# 8) EVALUATE
# -------------------------
def eval_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Some models may not support predict_proba (ours do)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n==============================")
    print(f"{name} Results")
    print("==============================")
    print("Accuracy :", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall   :", round(rec, 4))
    print("F1 Score :", round(f1, 4))

    if y_prob is not None:
        auc = roc_auc_score(y_test, y_prob)
        print("ROC-AUC  :", round(auc, 4))

    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

eval_model("Logistic Regression (Pipeline)", lr_model, X_test, y_test)
eval_model("Random Forest (Pipeline)", rf_model, X_test, y_test)

# -------------------------
# 9) SAVE MODELS FOR BACKEND
# -------------------------
joblib.dump(lr_model, "lr_model.pkl")
joblib.dump(rf_model, "rf_model.pkl")

# Also save the original feature column order (very useful for frontend/backend)
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("\nSaved:")
print("- lr_model.pkl")
print("- rf_model.pkl")
print("- feature_columns.pkl")

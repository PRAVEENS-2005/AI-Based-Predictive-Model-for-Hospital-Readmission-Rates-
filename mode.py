import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)

ZIP_PATH = "diabetes+130-us+hospitals+for+years+1999-2008.zip"

# 1. Load data
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    with z.open("diabetic_data.csv") as f:
        df = pd.read_csv(f)

# 2. Clean data
df = df.replace("?", np.nan)
df = df.drop(columns=["weight", "payer_code", "medical_specialty"], errors="ignore")

# 3. Create binary target
# 1 = readmitted within 30 days, 0 = otherwise
df["target"] = (df["readmitted"] == "<30").astype(int)

# 4. Remove leakage and ID columns
X = df.drop(columns=["readmitted", "target", "encounter_id", "patient_nbr"], errors="ignore")
y = df["target"]

# 5. Column groups
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# 6. Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols),
])

# 7. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, solver="liblinear")),
])

model.fit(X_train, y_train)

# 9. Predict
y_prob = model.predict_proba(X_test)[:,1]
y_pred = (y_prob >= 0.5).astype(int)

# 10. Metrics
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy :", round(acc, 4))
print("ROC-AUC  :", round(auc, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, digits=4))
print("Confusion Matrix:\n", cm)

# 11. Confusion matrix plot
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not <30", "<30"]).plot(cmap="Blues")
plt.title("Confusion Matrix - 30 Day Readmission")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.close()

# 12. ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - 30 Day Readmission")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=200)
plt.close()

# 13. Explainability: coefficients
X_train_processed = model.named_steps["preprocessor"].transform(X_train)
X_test_processed = model.named_steps["preprocessor"].transform(X_test)

ohe = model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = ohe.get_feature_names_out(categorical_cols)
all_feature_names = numerical_cols + list(cat_feature_names)

clf = model.named_steps["classifier"]
coefficients = clf.coef_[0]

feature_importance = pd.DataFrame({
    "Feature": all_feature_names,
    "Coefficient": coefficients,
})
feature_importance["AbsCoefficient"] = feature_importance["Coefficient"].abs()
feature_importance = feature_importance.sort_values(by="AbsCoefficient", ascending=False)
feature_importance.to_csv("top_features.csv", index=False)

# Plot top 20
top20 = feature_importance.head(20).sort_values(by="Coefficient")
plt.figure(figsize=(10, 8))
plt.barh(top20["Feature"], top20["Coefficient"])
plt.xlabel("Coefficient")
plt.title("Top 20 Logistic Regression Features")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=200)
plt.close()

# 14. SHAP summary (optional if installed)
try:
    explainer = shap.LinearExplainer(clf, X_train_processed)
    shap_values = explainer.shap_values(X_test_processed)

    shap.summary_plot(
        shap_values,
        X_test_processed,
        feature_names=all_feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 15. Print top 3 features and their values causing readmission for one patient
    sample_index = 100  # change patient index if needed

    sample_shap = shap_values[sample_index]
    sample_data = X_test_processed[sample_index]

    if hasattr(sample_data, "toarray"):
        sample_data = sample_data.toarray().flatten()
    else:
        sample_data = np.array(sample_data).flatten()

    explain_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Feature_Value": sample_data,
        "SHAP_Value": sample_shap
    })

    # keep only features increasing readmission risk
    positive_features = explain_df[explain_df["SHAP_Value"] > 0]

    # sort by strongest contribution
    top3 = positive_features.sort_values(by="SHAP_Value", ascending=False).head(3)

    print("\nTop 3 features causing readmission for patient", sample_index)
    print(top3.to_string(index=False))

    print("\nActual class:", y_test.iloc[sample_index])
    print("Predicted class:", y_pred[sample_index])
    print("Predicted probability:", round(y_prob[sample_index], 4))

except Exception as e:
    print("SHAP plot could not be generated:", e)

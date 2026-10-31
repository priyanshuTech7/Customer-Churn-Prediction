from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PATH  = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR  = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "customer_churn_model.pkl"


print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)


df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df = df.drop(columns=["customerID"])


X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected values in Churn column.")

numeric_features     = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("Numeric features    :", numeric_features)
print("Categorical features:", categorical_features)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}")


preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(),                       numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])


CANDIDATES = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, max_depth=12, random_state=42, class_weight="balanced", n_jobs=-1
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=7, random_state=42, class_weight="balanced"
    ),
}

print("\n" + "=" * 52)
print("  Model comparison")
print("=" * 52)

comparison_results: dict = {}

for name, clf in CANDIDATES.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]

    comparison_results[name] = {
        "accuracy":  float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall":    float(recall_score(y_test, preds, zero_division=0)),
        "f1_score":  float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_test, probs)),
    }
    r = comparison_results[name]
    print(
        f"  {name:<30}  Acc={r['accuracy']:.3f}  "
        f"Rec={r['recall']:.3f}  AUC={r['roc_auc']:.4f}"
    )


print("\nTraining final model (Balanced Logistic Regression)...")
final_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    )),
])
final_model.fit(X_train, y_train)


predictions  = final_model.predict(X_test)
probabilities = final_model.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall    = recall_score(y_test, predictions, zero_division=0)
f1        = f1_score(y_test, predictions, zero_division=0)
roc_auc   = roc_auc_score(y_test, probabilities)
cm        = confusion_matrix(y_test, predictions)
fpr, tpr, _ = roc_curve(y_test, probabilities)

print("\nFinal model — Balanced Logistic Regression")
print("-" * 44)
print(f"Accuracy  : {accuracy  * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall    * 100:.2f}%")
print(f"F1 Score  : {f1        * 100:.2f}%")
print(f"ROC-AUC   : {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=["No Churn", "Churn"]))
print("Confusion Matrix:\n", cm)


MODEL_DIR.mkdir(parents=True, exist_ok=True)

metrics = {
    "accuracy":  float(accuracy),
    "precision": float(precision),
    "recall":    float(recall),
    "f1_score":  float(f1),
    "roc_auc":   float(roc_auc),
}

final_model.evaluation_metrics_ = metrics
final_model.confusion_matrix_   = cm.tolist()
final_model.roc_curve_          = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
final_model.model_comparison_   = comparison_results
final_model.test_size_          = int(len(y_test))

joblib.dump(final_model, MODEL_PATH)

print("\n[SUCCESS] Model saved successfully!")
print("  Location :", MODEL_PATH)
print("  Metrics  :", metrics)
print("  Comparison models:", list(comparison_results.keys()))

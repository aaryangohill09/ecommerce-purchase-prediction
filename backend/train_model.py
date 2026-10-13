
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from preprocessing import preprocess_and_save

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "backend" / "model"
MODEL_PATH = MODEL_DIR / "purchase_model.pkl"
METRICS_PATH = BASE_DIR / "reports" / "model_metrics.json"

TARGET = "purchase"


def build_preprocessor(numeric_features, categorical_features):
    transformers = []

    if numeric_features:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(
            ("numeric", numeric_pipe, numeric_features)
        )

    if categorical_features:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(
            ("categorical", categorical_pipe, categorical_features)
        )

    return ColumnTransformer(transformers)


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(
            precision_score(y_test, pred, zero_division=0)
        ), 4),
        "recall": round(float(
            recall_score(y_test, pred, zero_division=0)
        ), 4),
        "f1": round(float(
            f1_score(y_test, pred, zero_division=0)
        ), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
    }


def main():
    cleaned = preprocess_and_save()

    if TARGET not in cleaned.columns:
        raise ValueError("Purchase target column is missing.")

    X = cleaned.drop(columns=[TARGET]).copy()
    y = cleaned[TARGET].astype(int)

    if y.nunique() != 2:
        raise ValueError("Purchase target must contain both classes: 0 and 1.")

    numeric_features = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_features = [
        col for col in X.columns
        if col not in numeric_features
    ]

    if X.empty or not len(X.columns):
        raise ValueError("No usable feature columns found.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42,
        ),
    }

    results = {}
    fitted = {}

    for name, estimator in candidates.items():
        pipe = Pipeline([
            (
                "preprocessor",
                build_preprocessor(
                    numeric_features,
                    categorical_features,
                ),
            ),
            ("model", estimator),
        ])

        pipe.fit(X_train, y_train)
        results[name] = evaluate(pipe, X_test, y_test)
        fitted[name] = pipe

        print(f"\n{name}: {results[name]}")

    best_name = max(
        results,
        key=lambda name: results[name]["roc_auc"],
    )
    best_model = fitted[best_name]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps({
            "dataset": "UCI Online Shoppers Purchasing Intention",
            "best_model": best_name,
            "models": results,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "features": X.columns.tolist(),
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
        }, indent=2),
        encoding="utf-8",
    )

    print(f"\nBest model: {best_name}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()

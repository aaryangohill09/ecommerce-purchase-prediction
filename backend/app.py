from pathlib import Path
import json
import joblib
import pandas as pd

from flask import Flask, jsonify, request
from flask_cors import CORS

# ==================================================
# APP CONFIGURATION
# ==================================================

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "cleaned_ecommerce_data.csv"
)

# Your project stores data in the root-level data folder.
if not DATA_PATH.exists():
    DATA_PATH = (
        PROJECT_DIR
        / "data"
        / "processed"
        / "cleaned_ecommerce_data.csv"
    )

MODEL_PATH = BASE_DIR / "model" / "purchase_model.pkl"

if not MODEL_PATH.exists():
    MODEL_PATH = PROJECT_DIR / "backend" / "model" / "purchase_model.pkl"

METRICS_PATH = PROJECT_DIR / "reports" / "model_metrics.json"


# ==================================================
# LOAD DATA AND MODEL
# ==================================================

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}. "
            "Run preprocessing.py first."
        )

    df = pd.read_csv(DATA_PATH)

    if "purchase" not in df.columns:
        raise ValueError(
            "The cleaned dataset must contain a 'purchase' column."
        )

    return df


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. "
            "Run train_model.py first."
        )

    return joblib.load(MODEL_PATH)


def json_safe(value):
    """Convert NumPy/Pandas values to JSON-friendly Python values."""
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def purchase_rate_by_category(df, column):
    """Calculate purchase rate (%) for each category."""
    if column not in df.columns:
        return []

    grouped = (
        df.groupby(column, dropna=False)["purchase"]
        .agg(["mean", "count"])
        .reset_index()
    )

    results = []

    for _, row in grouped.iterrows():
        category = row[column]

        if pd.isna(category):
            category = "Unknown"

        results.append({
            "category": str(category),
            "purchase_rate": round(float(row["mean"]) * 100, 2),
            "sessions": int(row["count"])
        })

    return results


# ==================================================
# HEALTH CHECK
# ==================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "E-Commerce Purchase Prediction API is running."
    })


@app.route("/health", methods=["GET"])
def health():
    try:
        df = load_data()
        model_exists = MODEL_PATH.exists()

        return jsonify({
            "status": "ok",
            "data_ready": not df.empty,
            "model_ready": model_exists,
            "rows": int(len(df))
        })

    except Exception as error:
        return jsonify({
            "status": "error",
            "error": str(error)
        }), 500


# ==================================================
# ANALYTICS DASHBOARD
# ==================================================

@app.route("/analytics", methods=["GET"])
def analytics():
    try:
        df = load_data()

        total_sessions = int(len(df))
        total_purchases = int(df["purchase"].sum())

        purchase_rate = (
            round((total_purchases / total_sessions) * 100, 2)
            if total_sessions else 0
        )

        # UCI dataset features
        avg_session_duration = None
        avg_pages_viewed = None

        duration_columns = [
            col for col in [
                "Administrative_Duration",
                "Informational_Duration",
                "ProductRelated_Duration"
            ]
            if col in df.columns
        ]

        page_columns = [
            col for col in [
                "Administrative",
                "Informational",
                "ProductRelated"
            ]
            if col in df.columns
        ]

        # Average total time across page categories
        if duration_columns:
            avg_session_duration = round(
                float(df[duration_columns].sum(axis=1).mean()), 2
            )

        # Average total pages viewed across page categories
        if page_columns:
            avg_pages_viewed = round(
                float(df[page_columns].sum(axis=1).mean()), 2
            )

        # Purchase rates by visitor type and month
        visitor_rates = purchase_rate_by_category(
            df, "VisitorType"
        )

        month_rates = purchase_rate_by_category(
            df, "Month"
        )

        # Product-related pages vs product-related duration
        session_pages_data = []

        if (
            "ProductRelated" in df.columns
            and "ProductRelated_Duration" in df.columns
        ):
            scatter_df = df[
                ["ProductRelated", "ProductRelated_Duration"]
            ].dropna()

            # Limit points for a responsive dashboard
            scatter_df = scatter_df.head(1000)

            session_pages_data = [
                {
                    "x": float(row["ProductRelated"]),
                    "y": float(row["ProductRelated_Duration"])
                }
                for _, row in scatter_df.iterrows()
            ]

        return jsonify({
            "kpis": {
                "total_sessions": total_sessions,
                "total_purchases": total_purchases,
                "purchase_rate": purchase_rate,
                "average_session_duration": avg_session_duration,
                "average_pages_viewed": avg_pages_viewed
            },
            "category_purchase_rates": {
                "visitor_purchase_rate": visitor_rates,
                "month_purchase_rate": month_rates
            },
            "session_pages_data": session_pages_data
        })

    except Exception as error:
        app.logger.exception("Analytics endpoint failed")

        return jsonify({
            "error": str(error)
        }), 500


# ==================================================
# PURCHASE PREDICTION
# ==================================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        model = load_model()
        df = load_data()

        payload = request.get_json(silent=True)

        if not isinstance(payload, dict):
            return jsonify({
                "error": "Please send prediction inputs as JSON."
            }), 400

        feature_columns = [
            column for column in df.columns
            if column != "purchase"
        ]

        missing_fields = [
            column for column in feature_columns
            if column not in payload
        ]

        if missing_fields:
            return jsonify({
                "error": "Some required input fields are missing.",
                "required_fields": missing_fields
            }), 400

        input_data = {}

        for column in feature_columns:
            value = payload[column]

            if value is None or value == "":
                return jsonify({
                    "error": f"Please provide a value for {column}."
                }), 400

            input_data[column] = value

        input_df = pd.DataFrame(
            [input_data],
            columns=feature_columns
        )

        # Normalize boolean values for the Weekend feature
        if "Weekend" in input_df.columns:
            if isinstance(input_df.loc[0, "Weekend"], str):
                input_df.loc[0, "Weekend"] = (
                    input_df.loc[0, "Weekend"].strip().lower()
                    == "true"
                )

        probability_values = model.predict_proba(input_df)[0]

        classes = list(model.classes_)
        positive_index = classes.index(1)

        purchase_probability = float(
            probability_values[positive_index] * 100
        )

        prediction = int(
            model.predict(input_df)[0]
        )

        label = (
            "Likely to Purchase"
            if prediction == 1
            else "Unlikely to Purchase"
        )

        return jsonify({
            "prediction": prediction,
            "label": label,
            "purchase_probability": round(
                purchase_probability, 2
            )
        })

    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 500

    except Exception as error:
        app.logger.exception("Prediction endpoint failed")

        return jsonify({
            "error": str(error)
        }), 500


# ==================================================
# MODEL METRICS
# ==================================================

@app.route("/metrics", methods=["GET"])
def metrics():
    try:
        if not METRICS_PATH.exists():
            return jsonify({
                "error": "Model metrics not found. Train the model first."
            }), 404

        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metrics_data = json.load(file)

        return jsonify(metrics_data)

    except Exception as error:
        app.logger.exception("Metrics endpoint failed")

        return jsonify({
            "error": str(error)
        }), 500


# ==================================================
# RUN SERVER
# ==================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
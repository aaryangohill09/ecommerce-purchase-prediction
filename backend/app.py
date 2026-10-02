from pathlib import Path

import joblib
import pandas as pd

from flask import Flask, jsonify, request
from flask_cors import CORS


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR.parent
    / "data"
    / "ecommerce_customer_data.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "purchase_model.pkl"
)


app = Flask(__name__)
CORS(app)


FEATURE_COLUMNS = [
    "device_type",
    "pages_viewed",
    "session_duration",
    "traffic_source",
    "previous_purchases"
]


def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run train_model.py first."
        )

    return joblib.load(MODEL_PATH)


def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found."
        )

    return pd.read_csv(DATA_PATH)


@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "project": "E-Commerce Customer Behavior & Purchase Prediction",
        "backend": "Python Flask",
        "status": "Running"
    })


@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy"
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Request body is required."
            }), 400

        missing_fields = [
            field
            for field in FEATURE_COLUMNS
            if field not in data
        ]

        if missing_fields:
            return jsonify({
                "error": "Missing fields",
                "fields": missing_fields
            }), 400

        input_data = pd.DataFrame(
            [{
                "device_type": data["device_type"],
                "pages_viewed": float(
                    data["pages_viewed"]
                ),
                "session_duration": float(
                    data["session_duration"]
                ),
                "traffic_source": data["traffic_source"],
                "previous_purchases": int(
                    data["previous_purchases"]
                )
            }]
        )

        model = load_model()

        prediction = int(
            model.predict(input_data)[0]
        )

        probability = float(
            model.predict_proba(input_data)[0][1]
        )

        return jsonify({
            "prediction": prediction,
            "purchase_probability": round(
                probability,
                4
            ),
            "result": (
                "Purchase Likely"
                if prediction == 1
                else "Purchase Unlikely"
            )
        })

    except ValueError as error:

        return jsonify({
            "error": str(error)
        }), 400

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500


@app.route("/analytics", methods=["GET"])
def analytics():

    try:

        df = load_data()

        total_sessions = len(df)

        purchase_rate = (
            df["purchase"].mean() * 100
        )

        average_session_duration = (
            df["session_duration"].mean()
        )

        average_pages = (
            df["pages_viewed"].mean()
        )

        device_analysis = (
            df.groupby("device_type")["purchase"]
            .mean()
            .mul(100)
            .round(2)
            .to_dict()
        )

        traffic_analysis = (
            df.groupby("traffic_source")["purchase"]
            .mean()
            .mul(100)
            .round(2)
            .to_dict()
        )

        scatter_data = (
            df[
                [
                    "session_duration",
                    "pages_viewed",
                    "purchase"
                ]
            ]
            .dropna()
            .head(500)
            .to_dict(orient="records")
        )

        return jsonify({

            "kpis": {
                "total_sessions": total_sessions,
                "purchase_rate": round(
                    purchase_rate,
                    2
                ),
                "average_session_duration": round(
                    average_session_duration,
                    2
                ),
                "average_pages_viewed": round(
                    average_pages,
                    2
                )
            },

            "device_purchase_rate":
                device_analysis,

            "traffic_purchase_rate":
                traffic_analysis,

            "session_pages_data":
                scatter_data
        })

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
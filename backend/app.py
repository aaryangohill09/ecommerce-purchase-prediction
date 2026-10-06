from pathlib import Path
import json
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_ecommerce_data.csv"
MODEL_PATH = BASE_DIR / "backend" / "model" / "purchase_model.pkl"

app = Flask(__name__)
CORS(app)

FEATURES = [
    "device_type",
    "pages_viewed",
    "session_duration",
    "traffic_source",
    "previous_purchases",
]

def load_data():
    return pd.read_csv(DATA_PATH)

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run: python train_model.py"
        )
    return joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return jsonify({
        "project": "E-Commerce Customer Behavior & Purchase Prediction",
        "status": "running",
        "endpoints": ["/health", "/analytics", "/predict"]
    })

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_ready": MODEL_PATH.exists(),
        "data_ready": DATA_PATH.exists(),
    })

@app.get("/analytics")
def analytics():
    try:
        df = load_data()

        device = (
            df.groupby("device_type")["purchase"]
            .mean()
            .mul(100)
            .round(2)
            .reset_index()
            .rename(columns={"purchase": "purchase_rate"})
        )

        traffic = (
            df.groupby("traffic_source")["purchase"]
            .mean()
            .mul(100)
            .round(2)
            .reset_index()
            .rename(columns={"purchase": "purchase_rate"})
        )

        scatter = df[["pages_viewed", "session_duration", "purchase"]].head(500).to_dict("records")

        return jsonify({
            "kpis": {
                "total_sessions": int(len(df)),
                "purchase_rate": round(float(df["purchase"].mean() * 100), 2),
                "average_session_duration": round(float(df["session_duration"].mean()), 2),
                "average_pages_viewed": round(float(df["pages_viewed"].mean()), 2),
            },
            "device_purchase_rate": device.to_dict("records"),
            "traffic_purchase_rate": traffic.to_dict("records"),
            "session_pages_data": scatter,
        })
    except Exception as error:
        return jsonify({"error": str(error)}), 500

@app.post("/predict")
def predict():
    try:
        payload = request.get_json(force=True)

        row = {
            "device_type": payload.get("device_type"),
            "pages_viewed": float(payload.get("pages_viewed")),
            "session_duration": float(payload.get("session_duration")),
            "traffic_source": payload.get("traffic_source"),
            "previous_purchases": float(payload.get("previous_purchases")),
        }

        if not row["device_type"] or not row["traffic_source"]:
            return jsonify({"error": "Device type and traffic source are required."}), 400

        if row["pages_viewed"] < 0 or row["session_duration"] < 0 or row["previous_purchases"] < 0:
            return jsonify({"error": "Numeric values cannot be negative."}), 400

        model = load_model()
        X = pd.DataFrame([row], columns=FEATURES)
        probability = float(model.predict_proba(X)[0][1])
        prediction = int(probability >= 0.5)

        return jsonify({
            "prediction": prediction,
            "purchase_probability": round(probability * 100, 2),
            "label": "Purchase Likely" if prediction else "Purchase Unlikely"
        })

    except (TypeError, ValueError):
        return jsonify({"error": "Please enter valid numeric values."}), 400
    except Exception as error:
        return jsonify({"error": str(error)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

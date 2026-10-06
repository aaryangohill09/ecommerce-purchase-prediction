# E-Commerce Customer Behavior & Purchase Prediction

Internship-level data science project that analyzes e-commerce customer sessions and predicts purchase probability.

## Project objectives

- Transform raw session data into a usable format
- Clean and validate customer behavior data
- Perform exploratory data analysis
- Identify purchase-related patterns
- Engineer predictive features
- Compare classification models
- Evaluate models using Accuracy, Precision, Recall, F1 and ROC-AUC
- Serve purchase probability through a Flask API
- Present results through a web dashboard
- Interpret findings from a business perspective

## Dataset

The repository contains a small **demo CSV** so the project runs immediately.

Before final internship submission, replace:

`data/raw/ecommerce_customer_data.csv`

with the actual dataset supplied by the internship. The expected logical fields are:

- device_type
- pages_viewed
- session_duration
- traffic_source
- previous_purchases
- purchase

`preprocessing.py` also accepts several common alternative column names.

## Setup

Recommended: Python 3.12 or 3.13.

```powershell
cd backend
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Train the models:

```powershell
python train_model.py
```

Generate EDA figures:

```powershell
python analysis.py
```

Start the API:

```powershell
python app.py
```

In a second terminal:

```powershell
cd frontend
python -m http.server 5500
```

Open:

`http://127.0.0.1:5500`

## API

### Health
`GET /health`

### Analytics
`GET /analytics`

### Prediction
`POST /predict`

Example:

```json
{
  "device_type": "Desktop",
  "pages_viewed": 10,
  "session_duration": 20,
  "traffic_source": "Organic",
  "previous_purchases": 2
}
```

## Machine learning

The training script compares:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

The model with the highest ROC-AUC on the held-out test set is saved as:

`backend/model/purchase_model.pkl`

## Important

Do not submit the demo metrics as internship results. Train the model again after replacing the demo CSV with the actual internship dataset.

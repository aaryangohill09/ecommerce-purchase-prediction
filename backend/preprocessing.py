from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "ecommerce_customer_data.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_ecommerce_data.csv"

COLUMN_ALIASES = {
    "device_type": ["device_type", "device", "device category", "device_category"],
    "pages_viewed": ["pages_viewed", "pages", "page_views", "pages viewed"],
    "session_duration": ["session_duration", "duration", "session duration", "session_duration_minutes"],
    "traffic_source": ["traffic_source", "source", "traffic source", "channel"],
    "previous_purchases": ["previous_purchases", "previous purchases", "past_purchases", "purchase_history"],
    "purchase": ["purchase", "purchased", "purchase_status", "purchase status", "made_purchase", "converted"],
}

def _normalise(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")

def load_raw_data(path=RAW_DATA_PATH):
    return pd.read_csv(path)

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = {_normalise(c): c for c in df.columns}
    rename = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            key = _normalise(alias)
            if key in normalized:
                rename[normalized[key]] = canonical
                break
    out = df.rename(columns=rename).copy()
    return out

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)

    required = list(COLUMN_ALIASES.keys())
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing) +
            ". Expected fields include device, pages viewed, session duration, "
            "traffic source, previous purchases and purchase status."
        )

    for col in ["pages_viewed", "session_duration", "previous_purchases"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["purchase"] = df["purchase"].astype(str).str.strip().str.lower().map({
        "1": 1, "0": 0, "true": 1, "false": 0,
        "yes": 1, "no": 0, "y": 1, "n": 0,
        "purchased": 1, "not purchased": 0
    })
    df["purchase"] = pd.to_numeric(df["purchase"], errors="coerce")

    df["device_type"] = df["device_type"].astype("string").str.strip().str.title()
    df["traffic_source"] = df["traffic_source"].astype("string").str.strip().str.title()

    df = df.drop_duplicates().copy()
    df = df.dropna(subset=required).copy()

    for col in ["pages_viewed", "session_duration", "previous_purchases"]:
        df = df[df[col] >= 0]

    df["purchase"] = df["purchase"].astype(int)
    df = df[df["purchase"].isin([0, 1])].copy()

    return df[required]

def preprocess_and_save():
    raw = load_raw_data()
    cleaned = clean_data(raw)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Raw rows: {len(raw)}")
    print(f"Clean rows: {len(cleaned)}")
    print(f"Saved: {PROCESSED_DATA_PATH}")
    return cleaned

if __name__ == "__main__":
    preprocess_and_save()

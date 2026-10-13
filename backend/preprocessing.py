
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    BASE_DIR / "data" / "raw" / "online_shoppers_intention.csv"
)
PROCESSED_DATA_PATH = (
    BASE_DIR / "data" / "processed" / "cleaned_ecommerce_data.csv"
)


def load_raw_data(path=RAW_DATA_PATH):
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardize column names
    df.columns = [
        str(col).strip().replace("-", "_").replace(" ", "_")
        for col in df.columns
    ]

    if "Revenue" not in df.columns:
        raise ValueError("Dataset must contain the Revenue target column.")

    # Convert the target into 0/1
    df["purchase"] = (
        df["Revenue"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0
        })
    )

    df = df.drop_duplicates().copy()
    df = df.dropna(subset=["purchase"]).copy()
    df["purchase"] = df["purchase"].astype(int)

    # Remove invalid numeric values without discarding
    # the dataset's other useful features
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=list(numeric_cols)).copy()

    return df.drop(columns=["Revenue"], errors="ignore")


def preprocess_and_save():
    raw = load_raw_data()
    cleaned = clean_data(raw)

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )
    cleaned.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Raw rows: {len(raw)}")
    print(f"Clean rows: {len(cleaned)}")
    print(f"Columns: {list(cleaned.columns)}")
    print(f"Purchase rate: {cleaned['purchase'].mean():.2%}")
    print(f"Saved: {PROCESSED_DATA_PATH}")

    return cleaned


if __name__ == "__main__":
    preprocess_and_save()


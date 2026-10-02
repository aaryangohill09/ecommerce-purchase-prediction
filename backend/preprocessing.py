import pandas as pd

REQUIRED_COLUMNS = [
    "device_type",
    "pages_viewed",
    "session_duration",
    "traffic_source",
    "previous_purchases",
    "purchase"
]


def load_dataset(file_path):
    df = pd.read_csv(file_path)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return df


def clean_dataset(df):
    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Numeric columns
    numeric_columns = [
        "pages_viewed",
        "session_duration",
        "previous_purchases",
        "purchase"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Categorical columns
    categorical_columns = [
        "device_type",
        "traffic_source"
    ]

    for column in categorical_columns:
        df[column] = df[column].astype(str).str.strip()

    # Remove rows where target is missing
    df = df.dropna(subset=["purchase"])

    # Keep valid target values
    df = df[df["purchase"].isin([0, 1])]

    return df


def prepare_features(df):
    feature_columns = [
        "device_type",
        "pages_viewed",
        "session_duration",
        "traffic_source",
        "previous_purchases"
    ]

    X = df[feature_columns]
    y = df["purchase"].astype(int)

    return X, y
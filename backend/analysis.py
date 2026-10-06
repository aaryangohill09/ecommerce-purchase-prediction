from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import preprocess_and_save

BASE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = BASE_DIR / "reports" / "figures"

def main():
    df = preprocess_and_save()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Purchase distribution
    df["purchase"].value_counts().sort_index().plot(kind="bar")
    plt.title("Purchase Distribution")
    plt.xlabel("Purchase")
    plt.ylabel("Sessions")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "purchase_distribution.png", dpi=180)
    plt.close()

    # Device purchase rate
    device = df.groupby("device_type")["purchase"].mean().sort_values(ascending=False)
    device.plot(kind="bar")
    plt.title("Purchase Rate by Device")
    plt.ylabel("Purchase Rate")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "purchase_rate_by_device.png", dpi=180)
    plt.close()

    # Traffic source purchase rate
    source = df.groupby("traffic_source")["purchase"].mean().sort_values(ascending=False)
    source.plot(kind="bar")
    plt.title("Purchase Rate by Traffic Source")
    plt.ylabel("Purchase Rate")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "purchase_rate_by_source.png", dpi=180)
    plt.close()

    # Session duration vs pages
    plt.scatter(df["pages_viewed"], df["session_duration"], alpha=0.45)
    plt.title("Pages Viewed vs Session Duration")
    plt.xlabel("Pages Viewed")
    plt.ylabel("Session Duration")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pages_vs_duration.png", dpi=180)
    plt.close()

    summary = {
        "total_sessions": int(len(df)),
        "purchase_rate": round(float(df["purchase"].mean()), 4),
        "average_session_duration": round(float(df["session_duration"].mean()), 2),
        "average_pages_viewed": round(float(df["pages_viewed"].mean()), 2),
        "device_purchase_rate": {
            str(k): round(float(v), 4) for k, v in device.items()
        },
        "traffic_purchase_rate": {
            str(k): round(float(v), 4) for k, v in source.items()
        },
    }
    (BASE_DIR / "reports" / "eda_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("EDA completed. Figures saved in reports/figures/")

if __name__ == "__main__":
    main()

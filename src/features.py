import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "booking_id",
    "no_show",
    "branch",
    "booking_month",
    "arrival_month",
    "arrival_day",
    "checkout_month",
    "checkout_day",
    "country",
    "first_time",
    "room",
    "price",
    "platform",
    "num_adults",
    "num_children",
}

def validate_columns(df: pd.DataFrame) -> None:
    missing = sorted(list(REQUIRED_COLUMNS - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def advanced_features(df):
    df = df.copy()

    # Weekend arrival
    df["weekend_arrival"] = df["arrival_day"].isin([6,7]).astype(int)

    # Peak season
    df["is_peak_season"] = df["arrival_month"].isin([6,7,12]).astype(int)

    # Price per guest
    df["price_per_guest"] = df["price_sgd"] / df["total_guests"].replace(0,1)

    # Booking window category
    df["lead_time_bucket"] = pd.cut(
        df["lead_time_days"],
        bins=[-1, 7, 30, 90, 365],
        labels=["last_minute","short","medium","long"]
    )

    # Family booking
    df["is_family"] = (df["num_children"] > 0).astype(int)

    # Interaction feature
    df["first_time_weekend"] = df["first_time"] * df["weekend_arrival"]

    #Long booking window + short stay is strong no-show indicator.
    df["booking_to_stay_ratio"] = df["lead_time_days"] / df["stay_length_days"].replace(0,1)

    return df


def clean_and_engineer(df, fx_usd_to_sgd=1.35):
    df = df.copy()

    # --- price parsing ---
    df["currency"] = df["price"].astype(str).str.extract(r"([A-Z]{3})", expand=False)

    price_numeric = (
        df["price"].astype(str)
        .str.replace(r"[A-Z]{3}\\$\\s*", "", regex=True)
        .replace("None", np.nan)
    )
    df["price_numeric"] = pd.to_numeric(price_numeric, errors="coerce")

    df["price_sgd"] = np.where(
        df["currency"].eq("USD"),
        df["price_numeric"] * fx_usd_to_sgd,
        df["price_numeric"]
    )

    # numeric coercion
    numeric_cols = [
        "booking_month","arrival_month","arrival_day",
        "checkout_month","checkout_day",
        "first_time","num_adults","num_children","no_show"
    ]

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # basic engineered features
    df["lead_time_days"] = (df["arrival_month"] - df["booking_month"]) * 30
    df["stay_length_days"] = (
        (df["checkout_month"] - df["arrival_month"]) * 30
        + (df["checkout_day"] - df["arrival_day"])
    ).abs()

    df["total_guests"] = df["num_adults"].fillna(0) + df["num_children"].fillna(0)

    # 🔥 Add advanced features here
    df = advanced_features(df)

    # Optional: drop raw noisy columns
    df = df.drop(columns=[
    "booking_id",
    "price",
    "arrival_day",
    "checkout_day",
    "arrival_month",
    "checkout_month",
    "booking_month"
])

    df = df.dropna(subset=["no_show"])
    df["no_show"] = df["no_show"].astype(int)

    return df
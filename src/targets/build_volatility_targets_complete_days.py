from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


INPUT_PATH = Path("data/processed/features/nifty_futures_features_complete_days.parquet")
OUTPUT_PATH = Path("data/processed/modeling_ready/nifty_futures_modeling_dataset_complete_days.parquet")

FUTURE_HORIZON = 25
LOW_Q = 0.30
HIGH_Q = 0.70


def add_future_realized_volatility(df: pd.DataFrame, horizon: int = FUTURE_HORIZON) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("timestamp_ist").reset_index(drop=True)

    future_rv = []
    log_returns = df["log_return_15m"].to_numpy()

    for i in range(len(df)):
        future_window = log_returns[i + 1:i + 1 + horizon]
        if len(future_window) < horizon or np.isnan(future_window).any():
            future_rv.append(np.nan)
        else:
            future_rv.append(np.sqrt(np.sum(future_window ** 2)))

    df["future_rv_1d"] = future_rv
    return df


def add_regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    valid_rv = df["future_rv_1d"].dropna()
    q_low = valid_rv.quantile(LOW_Q)
    q_high = valid_rv.quantile(HIGH_Q)

    def label_rv(x):
        if pd.isna(x):
            return None
        if x <= q_low:
            return "Low"
        elif x <= q_high:
            return "Medium"
        return "High"

    df["volatility_regime"] = df["future_rv_1d"].apply(label_rv)
    df["volatility_regime_code"] = df["volatility_regime"].map({"Low": 0, "Medium": 1, "High": 2})

    print(f"Quantile thresholds -> Low: <= {q_low:.6f}, High: > {q_high:.6f}")
    return df


def add_transition_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    valid_current = df["rv_1d"].dropna()
    cur_low = valid_current.quantile(LOW_Q)
    cur_high = valid_current.quantile(HIGH_Q)

    def current_label(x):
        if pd.isna(x):
            return None
        if x <= cur_low:
            return "Low"
        elif x <= cur_high:
            return "Medium"
        return "High"

    df["current_regime_proxy"] = df["rv_1d"].apply(current_label)

    df["transition_to_high"] = np.where(
        (df["current_regime_proxy"].isin(["Low", "Medium"])) &
        (df["volatility_regime"] == "High"),
        1,
        0,
    )

    df.loc[df["volatility_regime"].isna() | df["current_regime_proxy"].isna(), "transition_to_high"] = np.nan

    return df


def main() -> None:
    df = pd.read_parquet(INPUT_PATH)

    print("Input shape:", df.shape)

    df = add_future_realized_volatility(df)
    df = add_regime_labels(df)
    df = add_transition_target(df)

    print("Output shape:", df.shape)

    print("\nVolatility regime distribution:")
    print(df["volatility_regime"].value_counts(dropna=False).to_string())

    print("\nTransition-to-high distribution:")
    print(df["transition_to_high"].value_counts(dropna=False).to_string())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nSaved modeling dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

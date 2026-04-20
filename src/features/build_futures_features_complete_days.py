from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


INPUT_PATH = Path("data/interim/cleaned/nifty_continuous_futures_complete_days.parquet")
OUTPUT_PATH = Path("data/processed/features/nifty_futures_features_complete_days.parquet")


def add_basic_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("timestamp_ist").reset_index(drop=True)

    df["return_15m"] = df["close"].pct_change()
    df["log_return_15m"] = np.log(df["close"] / df["close"].shift(1))

    df["range_abs"] = df["high"] - df["low"]
    df["range_pct_close"] = df["range_abs"] / df["close"]
    df["body_abs"] = df["close"] - df["open"]
    df["body_pct_open"] = df["body_abs"] / df["open"]

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["return_1h"] = df["close"].pct_change(periods=4)
    df["return_halfday"] = df["close"].pct_change(periods=12)
    df["return_1d"] = df["close"].pct_change(periods=25)

    df["rv_1h"] = df["log_return_15m"].rolling(window=4).std()
    df["rv_halfday"] = df["log_return_15m"].rolling(window=12).std()
    df["rv_1d"] = df["log_return_15m"].rolling(window=25).std()

    df["close_ma_1h"] = df["close"].rolling(window=4).mean()
    df["close_ma_halfday"] = df["close"].rolling(window=12).mean()
    df["close_ma_1d"] = df["close"].rolling(window=25).mean()

    return df


def add_oi_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["oi_change"] = df["open_interest"].diff()
    df["oi_change_pct"] = df["open_interest"].pct_change()
    df["oi_change_1h"] = df["open_interest"].diff(periods=4)
    df["oi_change_1d"] = df["open_interest"].diff(periods=25)

    df["volume_change"] = df["volume"].diff()
    df["volume_change_pct"] = df["volume"].pct_change()
    df["volume_ma_1h"] = df["volume"].rolling(window=4).mean()
    df["volume_ma_1d"] = df["volume"].rolling(window=25).mean()

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    ts = pd.to_datetime(df["timestamp_ist"], errors="coerce")

    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["day_of_week"] = ts.dt.dayofweek
    df["opening_interval_flag"] = (df["trade_time"] == "09:15:00").astype(int)

    return df


def add_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")

    daily_last_close = df.groupby("trade_date")["close"].last().shift(1)
    daily_first_open = df.groupby("trade_date")["open"].first()

    gap_df = pd.DataFrame({
        "trade_date": daily_first_open.index,
        "prev_day_close": daily_last_close.values,
        "day_open": daily_first_open.values,
    })

    gap_df["gap_abs"] = gap_df["day_open"] - gap_df["prev_day_close"]
    gap_df["gap_pct"] = gap_df["gap_abs"] / gap_df["prev_day_close"]

    df = df.merge(gap_df[["trade_date", "gap_abs", "gap_pct"]], on="trade_date", how="left")

    df["gap_abs_firstbar"] = np.where(df["opening_interval_flag"] == 1, df["gap_abs"], np.nan)
    df["gap_pct_firstbar"] = np.where(df["opening_interval_flag"] == 1, df["gap_pct"], np.nan)

    return df


def main() -> None:
    df = pd.read_parquet(INPUT_PATH)

    print("Input shape:", df.shape)

    df = add_basic_price_features(df)
    df = add_rolling_features(df)
    df = add_oi_volume_features(df)
    df = add_time_features(df)
    df = add_gap_features(df)

    print("Output shape:", df.shape)
    print("\nSample rows:")
    print(
        df[
            [
                "timestamp_ist",
                "close",
                "return_15m",
                "log_return_15m",
                "rv_1h",
                "rv_1d",
                "oi_change",
                "volume_change",
                "opening_interval_flag",
            ]
        ].head(10).to_string(index=False)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nSaved features file to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

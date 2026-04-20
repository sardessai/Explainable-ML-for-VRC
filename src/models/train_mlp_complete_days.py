from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


INPUT_PATH = Path("data/processed/modeling_ready/nifty_futures_modeling_dataset_complete_days.parquet")
TARGET_COL = "volatility_regime_code"

EXCLUDE_COLS = {
    "timestamp",
    "timestamp_ist",
    "trade_date",
    "trade_time",
    "symbol_name",
    "source_file",
    "security_id",
    "expiry_date",
    "volatility_regime",
    "volatility_regime_code",
    "future_rv_1d",
    "transition_to_high",
    "current_regime_proxy",
}


def load_dataset() -> pd.DataFrame:
    return pd.read_parquet(INPUT_PATH)


def prepare_data(df: pd.DataFrame):
    df = df.copy()
    df = df[df[TARGET_COL].notna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in EXCLUDE_COLS]

    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    return X, y, feature_cols


def time_based_split(X, y, train_ratio: float = 0.8):
    split_idx = int(len(X) * train_ratio)
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def build_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=0.001,
                    batch_size=32,
                    learning_rate_init=0.001,
                    max_iter=500,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                ),
            ),
        ]
    )


def main():
    df = load_dataset()
    print("Loaded dataset shape:", df.shape)

    X, y, feature_cols = prepare_data(df)

    print("\nFeature matrix shape:", X.shape)
    print("Target shape:", y.shape)

    print("\nOverall target distribution:")
    print(y.value_counts().sort_index().to_string())

    X_train, X_test, y_train, y_test = time_based_split(X, y, train_ratio=0.8)

    print("\nTrain distribution:")
    print(y_train.value_counts().sort_index().to_string())

    print("\nTest distribution:")
    print(y_test.value_counts().sort_index().to_string())

    print("\nTrain/Test split:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nAccuracy:")
    print(accuracy_score(y_test, y_pred))

    print("\nBalanced Accuracy:")
    print(balanced_accuracy_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


INPUT_PATH = Path("data/processed/modeling_ready/nifty_futures_modeling_dataset.parquet")


TARGET_COL = "volatility_regime_code"

# Columns to exclude from features
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
    df = pd.read_parquet(INPUT_PATH)
    return df


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    # Keep only rows with valid target
    df = df[df[TARGET_COL].notna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    feature_cols = [col for col in numeric_cols if col not in EXCLUDE_COLS]

    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()

    return X, y


def time_based_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_idx = int(len(X) * train_ratio)

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()

    return X_train, X_test, y_train, y_test


def build_pipeline() -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
                ),
            ),
        ]
    )
    return pipeline


def main() -> None:
    df = load_dataset()
    print("Loaded dataset shape:", df.shape)

    X, y = prepare_data(df)

    print("\nFeature matrix shape:", X.shape)
    print("Target shape:", y.shape)
    print("\nTarget distribution:")
    print(y.value_counts().sort_index().to_string())

    print("\nFeature columns:")
    print(X.columns.tolist())

    X_train, X_test, y_train, y_test = time_based_split(X, y, train_ratio=0.8)

    print("\nTrain/Test split:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\nLogistic Regression Accuracy:")
    print(acc)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
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
    return X, y


def time_based_split(X, y, train_ratio: float = 0.8):
    split_idx = int(len(X) * train_ratio)
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }


def main():
    df = load_dataset()
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = time_based_split(X, y)

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, class_weight="balanced", random_state=42)),
        ]),
        "MLP": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                alpha=0.001,
                batch_size=32,
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
            )),
        ]),
    }

    results = []
    for name, model in models.items():
        results.append(evaluate_model(name, model, X_train, X_test, y_train, y_test))

    results_df = pd.DataFrame(results).sort_values(by="balanced_accuracy", ascending=False)

    print("\nModel comparison:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()

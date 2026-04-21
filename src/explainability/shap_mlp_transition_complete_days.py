from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import shap

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


INPUT_PATH = Path("data/processed/modeling_ready/nifty_futures_modeling_dataset_complete_days.parquet")
TARGET_COL = "transition_to_high"

GLOBAL_OUTPUT = Path("outputs/shap_outputs/mlp_transition_shap_global_importance.csv")
LOCAL_OUTPUT = Path("outputs/shap_outputs/mlp_transition_shap_local_values.csv")

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
    X, y, feature_cols = prepare_data(df)
    X_train, X_test, y_train, y_test = time_based_split(X, y, train_ratio=0.8)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]

    X_train_imp = imputer.transform(X_train)
    X_test_imp = imputer.transform(X_test)

    X_train_scaled = scaler.transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # Keep SHAP manageable
    background_size = min(100, len(X_train_scaled))
    explain_size = min(50, len(X_test_scaled))

    background = X_train_scaled[:background_size]
    explain_data = X_test_scaled[:explain_size]

    def predict_proba_fn(data):
        return model.predict_proba(data)

    print("Building SHAP KernelExplainer...")
    explainer = shap.KernelExplainer(predict_proba_fn, background)

    print("Computing SHAP values...")
    shap_values = explainer.shap_values(explain_data, nsamples=100)

    # Normalize SHAP output for binary class 1
    if isinstance(shap_values, list):
        class_1_shap = shap_values[1]
    else:
        shap_values = np.array(shap_values)

        if shap_values.ndim == 3:
            # shape = (n_samples, n_features, n_classes)
            class_1_shap = shap_values[:, :, 1]
        elif shap_values.ndim == 2:
            # shape = (n_samples, n_features)
            class_1_shap = shap_values
        else:
            raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}")

    print("Resolved SHAP shape for class 1:", class_1_shap.shape)

    explain_df = pd.DataFrame(explain_data, columns=feature_cols)
    shap_df = pd.DataFrame(class_1_shap, columns=feature_cols)

    global_importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": np.abs(class_1_shap).mean(axis=0),
    }).sort_values(by="mean_abs_shap", ascending=False)

    local_output = explain_df.copy()
    for col in feature_cols:
        local_output[f"shap_{col}"] = shap_df[col]

    GLOBAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    global_importance.to_csv(GLOBAL_OUTPUT, index=False)
    local_output.to_csv(LOCAL_OUTPUT, index=False)

    print("\nTop 15 SHAP global features:")
    print(global_importance.head(15).to_string(index=False))

    print(f"\nSaved global SHAP importance to: {GLOBAL_OUTPUT}")
    print(f"Saved local SHAP values to: {LOCAL_OUTPUT}")


if __name__ == "__main__":
    main()
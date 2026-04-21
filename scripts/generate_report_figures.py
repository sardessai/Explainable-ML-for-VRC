from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def figure_3_transition_target_distribution() -> None:
    labels = ["0", "1"]
    counts = [1135, 189]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, counts)
    plt.xlabel("Transition to High")
    plt.ylabel("Count")
    plt.title("Figure 3. Binary Transition Target Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_3_transition_target_distribution.png", dpi=300)
    plt.close()


def figure_5_transition_model_comparison() -> None:
    df = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "MLP"],
        "Balanced Accuracy": [0.6290, 0.5734, 0.7056],
        "Recall": [1.0000, 0.4318, 0.8636],
        "F1": [0.3492, 0.3016, 0.4176],
    })

    x = range(len(df))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar([i - width for i in x], df["Balanced Accuracy"], width=width, label="Balanced Accuracy")
    plt.bar(x, df["Recall"], width=width, label="Recall")
    plt.bar([i + width for i in x], df["F1"], width=width, label="F1")

    plt.xticks(list(x), df["Model"], rotation=15)
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Figure 5. Binary Transition Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_5_transition_model_comparison.png", dpi=300)
    plt.close()


def figure_8_shap_global_importance() -> None:
    df = pd.DataFrame({
        "feature": [
            "rv_1d",
            "volume_ma_1h",
            "gap_pct",
            "gap_abs",
            "rv_halfday",
            "return_halfday",
            "open_interest",
            "day_of_week",
            "return_1d",
            "volume_ma_1d",
        ],
        "mean_abs_shap": [
            0.153062,
            0.083576,
            0.070872,
            0.056165,
            0.048343,
            0.042376,
            0.041151,
            0.029841,
            0.028467,
            0.024298,
        ],
    }).sort_values("mean_abs_shap", ascending=True)

    plt.figure(figsize=(9, 6))
    plt.barh(df["feature"], df["mean_abs_shap"])
    plt.xlabel("Mean Absolute SHAP Value")
    plt.title("Figure 8. SHAP Global Importance for MLP Transition Model")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_8_shap_global_importance.png", dpi=300)
    plt.close()


def main() -> None:
    figure_3_transition_target_distribution()
    figure_5_transition_model_comparison()
    figure_8_shap_global_importance()

    print("Saved figures to:")
    print(OUTPUT_DIR / "figure_3_transition_target_distribution.png")
    print(OUTPUT_DIR / "figure_5_transition_model_comparison.png")
    print(OUTPUT_DIR / "figure_8_shap_global_importance.png")


if __name__ == "__main__":
    main()

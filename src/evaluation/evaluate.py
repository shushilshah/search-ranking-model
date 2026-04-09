"""
src/evaluation/evaluate.py

Loads a trained XGBoost model, runs evaluation on the test set,
plots feature importance, and optionally computes SHAP values.

Usage:
    python src/evaluation/evaluate.py
    python src/evaluation/evaluate.py --model outputs/model.json --shap
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

from src.data.loader import load_raw, query_group_split, to_xgb_arrays, FEATURE_COLS
from src.evaluation.metrics import compute_all_metrics


def plot_feature_importance(model: xgb.Booster, output_path: str = "outputs/feature_importance.png") -> None:
    importance = model.get_score(importance_type="gain")
    # Fill zeros for missing features
    feat_imp = {f: importance.get(f, 0.0) for f in FEATURE_COLS}
    sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([f[0] for f in sorted_feats], [f[1]
            for f in sorted_feats], color="#534AB7")
    ax.set_xlabel("Gain")
    ax.set_title("Feature importance (gain)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Feature importance plot → {output_path}")


def plot_shap(model: xgb.Booster, X: np.ndarray, output_path: str = "outputs/shap_summary.png") -> None:
    try:
        import shap
    except ImportError:
        print("shap not installed — skipping SHAP plot. Run: pip install shap")
        return

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(xgb.DMatrix(X, feature_names=FEATURE_COLS))

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.beeswarm(shap_values, max_display=12, show=False)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"SHAP summary plot → {output_path}")


def evaluate(
    model_path: str = "outputs/model.json",
    config_path: str = "configs/model_config.yaml",
    use_shap: bool = False,
) -> dict:
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    df = load_raw(data_cfg.get("features_path", data_cfg["raw_path"]))
    _, _, df_test = query_group_split(
        df,
        test_size=data_cfg["test_size"],
        val_size=data_cfg["val_size"],
        random_seed=data_cfg["random_seed"],
    )

    X_test, y_test, g_test = to_xgb_arrays(df_test)

    model = xgb.Booster()
    model.load_model(model_path)
    print(f"Loaded model from {model_path}")

    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=FEATURE_COLS)
    dtest.set_group(g_test)

    preds = model.predict(dtest)
    k_values = cfg["evaluation"]["k_values"]
    metrics = compute_all_metrics(preds, y_test, g_test, k_values)

    print("\n── Evaluation results ──")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")

    plot_feature_importance(model, cfg["output"]["feature_importance_plot"])

    if use_shap:
        plot_shap(model, X_test)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/model.json")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--shap", action="store_true")
    args = parser.parse_args()
    evaluate(args.model, args.config, args.shap)

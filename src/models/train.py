"""
src/models/train.py

Trains a Learning-to-Rank model using XGBoost.
Supports pointwise (regression), pairwise, and listwise (NDCG) objectives.

Usage:
    python src/models/train.py
    python src/models/train.py --config configs/model_config.yaml
    python src/models/train.py --mode pairwise
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
import yaml

from src.data.loader import load_raw, query_group_split, to_xgb_arrays, FEATURE_COLS
from src.evaluation.metrics import ndcg_at_k, mrr, mean_average_precision


OBJECTIVE_MAP = {
    "pointwise": "reg:squarederror",
    "pairwise":  "rank:pairwise",
    "listwise":  "rank:ndcg",
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_xgb_params(cfg: dict, mode: str | None = None) -> dict:
    m = cfg["model"]
    if mode:
        m["objective"] = OBJECTIVE_MAP[mode]

    return {
        "objective":        m.get("objective", "rank:ndcg"),
        "eval_metric":      m.get("eval_metric", "ndcg@10"),
        "learning_rate":    m.get("learning_rate", 0.05),
        "max_depth":        m.get("max_depth", 6),
        "min_child_weight": m.get("min_child_weight", 5),
        "subsample":        m.get("subsample", 0.8),
        "colsample_bytree": m.get("colsample_bytree", 0.8),
        "reg_alpha":        m.get("reg_alpha", 0.1),
        "reg_lambda":       m.get("reg_lambda", 1.0),
        "verbosity":        m.get("verbosity", 1),
        "seed":             cfg["data"].get("random_seed", 42),
    }


def train(config_path: str = "configs/model_config.yaml", mode: str | None = None):
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    # ── Load & split ──────────────────────────────────────────────────────────
    df = load_raw(data_cfg.get("features_path", data_cfg["raw_path"]))
    df_train, df_val, df_test = query_group_split(
        df,
        test_size=data_cfg["test_size"],
        val_size=data_cfg["val_size"],
        random_seed=data_cfg["random_seed"],
    )

    X_train, y_train, g_train = to_xgb_arrays(df_train)
    X_val,   y_val,   g_val = to_xgb_arrays(df_val)
    X_test,  y_test,  g_test = to_xgb_arrays(df_test)

    # ── XGBoost DMatrix ───────────────────────────────────────────────────────
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLS)
    dtrain.set_group(g_train)

    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_COLS)
    dval.set_group(g_val)

    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=FEATURE_COLS)
    dtest.set_group(g_test)

    # ── Train ─────────────────────────────────────────────────────────────────
    params = build_xgb_params(cfg, mode)
    print(f"\nTraining with objective: {params['objective']}")

    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=model_cfg.get("n_estimators", 500),
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=model_cfg.get("early_stopping_rounds", 30),
        evals_result=evals_result,
        verbose_eval=50,
    )

    # ── Evaluate on test set ───────────────────────────────────────────────────
    test_preds = model.predict(dtest)
    print("\n── Test set evaluation ──")
    report = evaluate_predictions(
        test_preds, y_test, g_test, cfg["evaluation"]["k_values"])

    # ── Save model & report ───────────────────────────────────────────────────
    # out_cfg = cfg["output"]
    # Path(out_cfg["model_path"]).parent.mkdir(parents=True, exist_ok=True)
    # model.save_model(out_cfg["model_path"])
    # print(f"\nModel saved → {out_cfg['model_path']}")

    # with open(out_cfg["report_path"], "w") as f:
    #     json.dump(report, f, indent=2)
    # print(f"Eval report → {out_cfg['report_path']}")

    # return model, report

    # ── Save model & report ───────────────────────────────────────────────────
    out_cfg = cfg["output"]

    # determine mode name
    mode_name = mode if mode else "listwise"

    # dynamic model path
    model_path = f"outputs/model_{mode_name}.json"
    report_path = f"outputs/eval_report_{mode_name}.json"

    Path("outputs").mkdir(parents=True, exist_ok=True)

    # save model
    model.save_model(model_path)
    print(f"\nModel saved → {model_path}")

    # add metadata (VERY useful later)
    report["model_type"] = mode_name
    report["objective"] = params["objective"]
    report["n_features"] = len(FEATURE_COLS)

    # save report
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Eval report → {report_path}")


def evaluate_predictions(preds: np.ndarray, labels: np.ndarray, groups: list, k_values: list) -> dict:
    report = {}
    offset = 0
    ndcg_scores = {k: [] for k in k_values}
    mrr_scores = []
    map_scores = []

    for g in groups:
        p = preds[offset: offset + g]
        l = labels[offset: offset + g]
        order = np.argsort(-p)  # descending by predicted score
        l_sorted = l[order]

        for k in k_values:
            ndcg_scores[k].append(ndcg_at_k(l_sorted, k))
        mrr_scores.append(mrr(l_sorted))
        map_scores.append(mean_average_precision(l_sorted))
        offset += g

    for k in k_values:
        report[f"ndcg@{k}"] = float(np.mean(ndcg_scores[k]))
        print(f"  NDCG@{k:2d}: {report[f'ndcg@{k}']:.4f}")

    report["mrr"] = float(np.mean(mrr_scores))
    report["map"] = float(np.mean(map_scores))
    print(f"  MRR:     {report['mrr']:.4f}")
    print(f"  MAP:     {report['map']:.4f}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument(
        "--mode", choices=["pointwise", "pairwise", "listwise"], default=None)
    args = parser.parse_args()
    train(args.config, args.mode)

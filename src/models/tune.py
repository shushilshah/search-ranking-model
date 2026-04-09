"""
src/models/tune.py

Hyperparameter tuning for the XGBoost LTR model using Optuna.
Optimises NDCG@10 on the validation set.

Usage:
    python src/models/tune.py --n-trials 50
"""

from __future__ import annotations
import argparse

import numpy as np
import optuna
import xgboost as xgb
import yaml

from src.data.loader import load_raw, query_group_split, to_xgb_arrays
from src.evaluation.metrics import ndcg_at_k

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial: optuna.Trial, dtrain, dval, y_val, g_val) -> float:
    params = {
        "objective":        "rank:ndcg",
        "eval_metric":      "ndcg@10",
        "verbosity":        0,
        "seed":             42,
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    n_estimators = trial.suggest_int("n_estimators", 100, 800)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dval, "val")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )

    preds = model.predict(dval)
    ndcg_scores = []
    offset = 0
    for g in g_val:
        p = preds[offset: offset + g]
        l = y_val[offset: offset + g]
        order = np.argsort(-p)
        ndcg_scores.append(ndcg_at_k(l[order], 10))
        offset += g

    return float(np.mean(ndcg_scores))


def tune(config_path: str = "configs/model_config.yaml", n_trials: int = 50) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    df = load_raw(data_cfg.get("features_path", data_cfg["raw_path"]))
    df_train, df_val, _ = query_group_split(
        df,
        test_size=data_cfg["test_size"],
        val_size=data_cfg["val_size"],
        random_seed=data_cfg["random_seed"],
    )

    from src.data.loader import FEATURE_COLS
    X_train, y_train, g_train = to_xgb_arrays(df_train)
    X_val,   y_val,   g_val = to_xgb_arrays(df_val)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLS)
    dtrain.set_group(g_train)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURE_COLS)
    dval.set_group(g_val)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, dtrain, dval, y_val, g_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    best["value"] = study.best_value
    print(f"\nBest NDCG@10: {study.best_value:.4f}")
    print(f"Best params:  {best}")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()
    tune(args.config, args.n_trials)

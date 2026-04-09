"""
Loads and splits query-document datasets for LTR training.
Produces query-grouped train/val/test splits (no query leakage).
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


FEATURE_COLS = [
    "bm25_score",
    "tfidf_cosine",
    "query_term_coverage",
    "title_match_score",
    "body_match_score",
    "doc_length",
    "doc_pagerank",
    "doc_freshness_days",
    "avg_click_rate",
    "query_length",
    "query_idf_sum",
    "is_navigational",
]

LABEL_COL = "relevance"
GROUP_COL = "qid"


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate(df)
    return df


def _validate(df: pd.DataFrame) -> None:
    required = set(FEATURE_COLS + [LABEL_COL, GROUP_COL])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def query_group_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split at the query level so no query appears in multiple splits.
    Returns (train_df, val_df, test_df).
    """
    queries = df[GROUP_COL].unique()

    # First split off test
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_seed)
    train_val_idx, test_idx = next(splitter.split(df, groups=df[GROUP_COL]))
    df_trainval = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # Then split val from trainval
    val_frac = val_size / (1 - test_size)
    splitter2 = GroupShuffleSplit(
        n_splits=1, test_size=val_frac, random_state=random_seed)
    train_idx, val_idx = next(splitter2.split(
        df_trainval, groups=df_trainval[GROUP_COL]))
    df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
    df_val = df_trainval.iloc[val_idx].reset_index(drop=True)

    print(
        f"Split sizes — train queries: {df_train[GROUP_COL].nunique()}, "
        f"val: {df_val[GROUP_COL].nunique()}, "
        f"test: {df_test[GROUP_COL].nunique()}"
    )
    return df_train, df_val, df_test


def to_xgb_arrays(df: pd.DataFrame):
    """
    Return (X, y, groups) ready for XGBoost DMatrix / sklearn API.
    groups = number of docs per query, in order (required by XGBoost ranker).
    """
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[LABEL_COL].values.astype(np.float32)
    groups = df.groupby(GROUP_COL, sort=False).size().values.tolist()
    return X, y, groups

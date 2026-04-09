"""
src/evaluation/metrics.py

Core IR evaluation metrics for Learning-to-Rank.

All functions accept a 1-D array of relevance labels ordered by the
predicted ranking (highest predicted score first).
"""

from __future__ import annotations
import numpy as np


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """Discounted Cumulative Gain @ k."""
    r = np.asarray(relevances[:k], dtype=float)
    if r.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, r.size + 2))
    return float(np.sum((2 ** r - 1) / discounts))


def ndcg_at_k(relevances: np.ndarray, k: int) -> float:
    """
    Normalised DCG @ k.
    relevances must already be sorted by predicted score (desc).
    """
    ideal = np.sort(relevances)[::-1]
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg_at_k(relevances, k) / ideal_dcg


def mrr(relevances: np.ndarray, threshold: float = 1.0) -> float:
    """
    Mean Reciprocal Rank.
    Returns 1/rank of the first relevant document (relevance >= threshold).
    """
    for i, r in enumerate(relevances):
        if r >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(relevances: np.ndarray, k: int, threshold: float = 1.0) -> float:
    """Precision @ k — fraction of top-k docs that are relevant."""
    top_k = np.asarray(relevances[:k])
    return float(np.mean(top_k >= threshold))


def mean_average_precision(relevances: np.ndarray, threshold: float = 1.0) -> float:
    """
    Average Precision for a single query.
    MAP is the mean of AP across queries.
    """
    hits = 0
    sum_precision = 0.0
    for i, r in enumerate(relevances):
        if r >= threshold:
            hits += 1
            sum_precision += hits / (i + 1)
    if hits == 0:
        return 0.0
    return sum_precision / hits


def compute_all_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    groups: list[int],
    k_values: list[int] = [1, 3, 5, 10],
) -> dict:
    """
    Compute all metrics over a full dataset given group sizes.

    Args:
        preds:  Model scores (higher = more relevant), shape (N,)
        labels: Ground-truth relevance labels, shape (N,)
        groups: List of query group sizes (sum == N)
        k_values: List of cutoffs for NDCG and P@k

    Returns:
        dict of metric_name → float
    """
    ndcg = {k: [] for k in k_values}
    prec = {k: [] for k in k_values}
    mrr_list, map_list = [], []

    offset = 0
    for g in groups:
        p = preds[offset: offset + g]
        l = labels[offset: offset + g]
        order = np.argsort(-p)
        l_sorted = l[order]

        for k in k_values:
            ndcg[k].append(ndcg_at_k(l_sorted, k))
            prec[k].append(precision_at_k(l_sorted, k))

        mrr_list.append(mrr(l_sorted))
        map_list.append(mean_average_precision(l_sorted))
        offset += g

    results: dict = {}
    for k in k_values:
        results[f"ndcg@{k}"] = float(np.mean(ndcg[k]))
        results[f"p@{k}"] = float(np.mean(prec[k]))
    results["mrr"] = float(np.mean(mrr_list))
    results["map"] = float(np.mean(map_list))
    return results

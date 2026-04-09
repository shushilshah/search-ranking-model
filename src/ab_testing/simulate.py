"""
src/ab_testing/simulate.py

Simulates realistic A/B experiment traffic for testing the analysis pipeline.
Generates impressions, clicks (with position bias), and conversions.

Usage:
    python src/ab_testing/simulate.py --n-users 500 --n-sessions 2000
"""

from __future__ import annotations
import argparse
import random
import time
import uuid

import numpy as np

from src.ab_testing.experiment import ExperimentRegistry
from src.ab_testing.tracker import EventStore, EventTracker

RNG = np.random.default_rng(42)

SAMPLE_QUERIES = [
    "machine learning tutorial", "python web scraping", "neural networks explained",
    "docker getting started", "sql joins guide", "git rebase vs merge",
    "rest api design", "linux commands cheatsheet", "kubernetes pods",
    "data pipeline etl",
]

N_CANDIDATE_DOCS = 10


def simulate_click(rank: int, treatment: bool) -> bool:
    """Position-biased click model. Treatment variant has higher CTR at top ranks."""
    # Cascade model: P(click | rank)
    base_ctr = 0.3 / rank ** 0.8
    boost = 0.05 if treatment else 0.0   # treatment is slightly better
    return RNG.uniform() < (base_ctr + boost)


def simulate_dwell(clicked: bool) -> float | None:
    if not clicked:
        return None
    return float(max(5.0, RNG.normal(45, 20)))


def run_simulation(
    n_users: int = 300,
    n_sessions: int = 1000,
    experiment_name: str = "exp_ltr_v2",
    tracker: EventTracker = None,
) -> dict:

    registry = ExperimentRegistry.from_yaml("configs/experiments.yaml")
    close_tracker = tracker is None
    if tracker is None:
        tracker = EventTracker("outputs/ab_events.jsonl")

    stats = {"impressions": 0, "clicks": 0, "conversions": 0}
    users = [f"u{i:05d}" for i in range(n_users)]

    for _ in range(n_sessions):
        user_id = random.choice(users)
        session_id = str(uuid.uuid4())[:8]
        query = random.choice(SAMPLE_QUERIES)

        variant = registry.assign(experiment_name, user_id)
        if variant is None:
            continue

        is_treatment = variant.name == "treatment"

        # Simulate n candidate doc IDs
        doc_ids = [f"doc_{i:04d}" for i in RNG.integers(
            0, 500, size=N_CANDIDATE_DOCS)]

        tracker.log_impression(
            user_id=user_id, experiment=experiment_name, variant=variant.name,
            query=query, session_id=session_id, doc_ids=doc_ids,
        )
        stats["impressions"] += 1

        # Simulate clicks (position-biased)
        for rank, doc_id in enumerate(doc_ids, start=1):
            clicked = simulate_click(rank, is_treatment)
            if clicked:
                dwell = simulate_dwell(True)
                tracker.log_click(
                    user_id=user_id, experiment=experiment_name, variant=variant.name,
                    query=query, doc_id=doc_id, rank=rank,
                    session_id=session_id, dwell_seconds=dwell,
                )
                stats["clicks"] += 1

                # Simulate conversion (low probability, higher for treatment)
                conv_prob = 0.08 if is_treatment else 0.05
                if RNG.uniform() < conv_prob:
                    tracker.log_conversion(
                        user_id=user_id, experiment=experiment_name, variant=variant.name,
                        query=query, doc_id=doc_id, session_id=session_id,
                    )
                    stats["conversions"] += 1

                break  # stop after first click (simplified cascade model)

    if close_tracker:
        tracker.close()

    print(f"Simulation complete: {stats}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-users", type=int, default=300)
    parser.add_argument("--n-sessions", type=int, default=1000)
    parser.add_argument("--experiment", default="exp_ltr_v2")
    args = parser.parse_args()

    run_simulation(args.n_users, args.n_sessions, args.experiment)

    # Run analysis immediately after simulation
    from src.ab_testing.analysis import ExperimentAnalyzer
    from src.ab_testing.tracker import EventStore

    store = EventStore("outputs/ab_events.db")
    store.ingest_jsonl("outputs/ab_events.jsonl")
    analyzer = ExperimentAnalyzer(store)
    report = analyzer.analyze(args.experiment)
    report.print_report()
    store.close()

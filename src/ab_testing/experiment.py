"""
src/ab_testing/experiment.py

Experiment registry and deterministic user-to-variant assignment.

Design:
  - Each experiment has a name, variants (with traffic splits), and a
    set of tracked metrics.
  - Assignment is deterministic: hash(user_id + experiment_name) → bucket
    so the same user always sees the same variant.
  - Experiments are stored in experiments.yaml and loaded at startup.

Usage:
    registry = ExperimentRegistry.from_yaml("configs/experiments.yaml")
    variant = registry.assign("exp_ltr_v2", user_id="u123")
    # → "control" or "treatment"
"""

from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class Variant:
    name: str
    traffic_fraction: float   # 0.0–1.0, all variants must sum to 1.0
    model_path: Optional[str] = None   # path to the ranking model for this variant
    description: str = ""


@dataclass
class Experiment:
    name: str
    variants: List[Variant]
    metrics: List[str]        # metric names to track, e.g. ["ndcg@10", "ctr", "mrr"]
    active: bool = True
    description: str = ""

    def __post_init__(self):
        total = sum(v.traffic_fraction for v in self.variants)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Experiment '{self.name}': variant traffic fractions must sum to 1.0, got {total:.4f}"
            )

    def assign(self, user_id: str) -> Variant:
        """
        Deterministically assign a user to a variant using a stable hash.
        Same user_id + experiment always maps to the same variant.
        """
        key = f"{self.name}:{user_id}"
        digest = hashlib.sha256(key.encode()).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF  # float in [0, 1)

        cumulative = 0.0
        for variant in self.variants:
            cumulative += variant.traffic_fraction
            if bucket < cumulative:
                return variant

        return self.variants[-1]  # safety fallback


class ExperimentRegistry:
    def __init__(self, experiments: Dict[str, Experiment]):
        self._experiments = experiments

    @classmethod
    def from_yaml(cls, path: str = "configs/experiments.yaml") -> "ExperimentRegistry":
        with open(path) as f:
            raw = yaml.safe_load(f)

        experiments = {}
        for exp_cfg in raw.get("experiments", []):
            variants = [
                Variant(
                    name=v["name"],
                    traffic_fraction=v["traffic_fraction"],
                    model_path=v.get("model_path"),
                    description=v.get("description", ""),
                )
                for v in exp_cfg["variants"]
            ]
            exp = Experiment(
                name=exp_cfg["name"],
                variants=variants,
                metrics=exp_cfg.get("metrics", []),
                active=exp_cfg.get("active", True),
                description=exp_cfg.get("description", ""),
            )
            experiments[exp.name] = exp

        print(f"Loaded {len(experiments)} experiments from {path}")
        return cls(experiments)

    def assign(self, experiment_name: str, user_id: str) -> Optional[Variant]:
        """Return the assigned variant, or None if the experiment is inactive/unknown."""
        exp = self._experiments.get(experiment_name)
        if exp is None:
            raise KeyError(f"Unknown experiment: '{experiment_name}'")
        if not exp.active:
            return None
        return exp.assign(user_id)

    def get(self, experiment_name: str) -> Optional[Experiment]:
        return self._experiments.get(experiment_name)

    def list_active(self) -> List[str]:
        return [name for name, exp in self._experiments.items() if exp.active]

    def __repr__(self) -> str:
        return f"ExperimentRegistry(experiments={list(self._experiments.keys())})"
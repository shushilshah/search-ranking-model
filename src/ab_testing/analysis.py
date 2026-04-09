"""
src/ab_testing/analysis.py

Statistical analysis for A/B experiment results.

Computes per-variant metrics and runs significance tests:
  - CTR (click-through rate)
  - MRR (mean reciprocal rank from click positions)
  - Conversion rate
  - Two-proportion z-test for CTR / conversion rate
  - Mann-Whitney U for continuous metrics (MRR, dwell time)

Usage:
    store = EventStore("outputs/ab_events.db")
    store.ingest_jsonl("outputs/ab_events.jsonl")
    results = ExperimentAnalyzer(store).analyze("exp_ltr_v2")
    results.print_report()
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

from src.ab_testing.tracker import EventStore


# ── Result containers ──────────────────────────────────────────────────────────

@dataclass
class VariantMetrics:
    variant: str
    n_impressions: int = 0
    n_clicks: int = 0
    n_conversions: int = 0
    ctr: float = 0.0               # clicks / impressions
    conversion_rate: float = 0.0   # conversions / impressions
    mrr: float = 0.0               # mean reciprocal rank of clicked docs
    avg_rank_clicked: float = 0.0  # mean rank position of clicked docs
    avg_dwell_seconds: float = 0.0


@dataclass
class ComparisonResult:
    metric: str
    control_value: float
    treatment_value: float
    relative_lift: float           # (treatment - control) / control
    p_value: float
    significant: bool              # p < alpha
    test_used: str
    confidence_interval: tuple = field(default_factory=tuple)

    @property
    def lift_pct(self) -> str:
        return f"{self.relative_lift * 100:+.2f}%"


@dataclass
class ExperimentReport:
    experiment: str
    control_variant: str
    treatment_variant: str
    variant_metrics: Dict[str, VariantMetrics]
    comparisons: List[ComparisonResult]
    sample_size_adequate: bool
    min_detectable_effect: float

    def print_report(self):
        print(f"\n{'='*60}")
        print(f"  Experiment: {self.experiment}")
        print(
            f"  Control: {self.control_variant}  |  Treatment: {self.treatment_variant}")
        print(f"{'='*60}")

        for vname, vm in self.variant_metrics.items():
            print(f"\n  [{vname}]")
            print(f"    Impressions : {vm.n_impressions:,}")
            print(f"    Clicks      : {vm.n_clicks:,}")
            print(f"    CTR         : {vm.ctr:.4f}  ({vm.ctr*100:.2f}%)")
            print(f"    Conversions : {vm.n_conversions:,}")
            print(f"    Conv. rate  : {vm.conversion_rate:.4f}")
            print(f"    MRR         : {vm.mrr:.4f}")
            print(f"    Avg rank    : {vm.avg_rank_clicked:.2f}")

        print(f"\n  {'─'*56}")
        print(
            f"  {'Metric':<22} {'Control':>9} {'Treatment':>10} {'Lift':>9} {'p-value':>9} {'Sig?':>6}")
        print(f"  {'─'*56}")

        for c in self.comparisons:
            sig = "✓" if c.significant else "✗"
            print(
                f"  {c.metric:<22} {c.control_value:>9.4f} {c.treatment_value:>10.4f} "
                f"{c.lift_pct:>9} {c.p_value:>9.4f} {sig:>6}"
            )

        print(
            f"\n  Sample size adequate: {'yes' if self.sample_size_adequate else 'no — collect more data'}")
        print(
            f"  Min detectable effect (CTR): ±{self.min_detectable_effect*100:.2f}%")
        print(f"{'='*60}\n")


# ── Analyzer ───────────────────────────────────────────────────────────────────

class ExperimentAnalyzer:

    def __init__(self, store: EventStore, alpha: float = 0.05):
        self.store = store
        self.alpha = alpha

    def _variant_metrics(self, experiment: str, variant: str) -> VariantMetrics:
        vm = VariantMetrics(variant=variant)

        imp = self.store.query(
            "SELECT COUNT(*) as n FROM impressions WHERE experiment=? AND variant=?",
            (experiment, variant)
        )
        vm.n_impressions = imp[0]["n"] if imp else 0

        clk = self.store.query(
            "SELECT COUNT(*) as n, AVG(rank) as avg_rank, AVG(dwell_seconds) as avg_dwell "
            "FROM clicks WHERE experiment=? AND variant=?",
            (experiment, variant)
        )
        if clk and clk[0]["n"]:
            vm.n_clicks = clk[0]["n"]
            vm.avg_rank_clicked = clk[0]["avg_rank"] or 0.0
            vm.avg_dwell_seconds = clk[0]["avg_dwell"] or 0.0

        conv = self.store.query(
            "SELECT COUNT(*) as n FROM conversions WHERE experiment=? AND variant=?",
            (experiment, variant)
        )
        vm.n_conversions = conv[0]["n"] if conv else 0

        if vm.n_impressions > 0:
            vm.ctr = vm.n_clicks / vm.n_impressions
            vm.conversion_rate = vm.n_conversions / vm.n_impressions

        # MRR from click rank positions
        ranks = self.store.query(
            "SELECT rank FROM clicks WHERE experiment=? AND variant=? AND rank > 0",
            (experiment, variant)
        )
        if ranks:
            # vm.mrr = float(np.mean([1.0 / r["rank"] for r in ranks]))
            vals = [1.0 / r["rank"] for r in ranks if r["rank"] > 0]
            vm.mrr = float(np.mean(vals)) if vals else 0.0

        return vm

    def _z_test_proportions(
        self,
        n_ctrl: int, k_ctrl: int,
        n_trt: int,  k_trt: int,
    ) -> tuple[float, tuple]:
        """Two-proportion z-test. Returns (p_value, 95% CI on difference)."""
        if n_ctrl == 0 or n_trt == 0:
            return 1.0, (0.0, 0.0)

        p1 = k_ctrl / n_ctrl
        p2 = k_trt / n_trt
        p_pool = (k_ctrl + k_trt) / (n_ctrl + n_trt)

        se_pool = math.sqrt(p_pool * (1 - p_pool) * (1/n_ctrl + 1/n_trt))
        if se_pool == 0:
            return 1.0, (0.0, 0.0)

        z = (p2 - p1) / se_pool
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        # 95% CI on the raw difference
        se_diff = math.sqrt(p1*(1-p1)/n_ctrl + p2*(1-p2)/n_trt)
        ci = (p2 - p1 - 1.96 * se_diff, p2 - p1 + 1.96 * se_diff)
        return p_value, ci

    def _mann_whitney(self, experiment: str, control: str, treatment: str, column: str, table: str):
        """Mann-Whitney U test on a continuous metric column."""
        ctrl_vals = self.store.query(
            f"SELECT {column} FROM {table} WHERE experiment=? AND variant=? AND {column} IS NOT NULL",
            (experiment, control)
        )
        trt_vals = self.store.query(
            f"SELECT {column} FROM {table} WHERE experiment=? AND variant=? AND {column} IS NOT NULL",
            (experiment, treatment)
        )
        a = [r[column] for r in ctrl_vals]
        b = [r[column] for r in trt_vals]

        if len(a) < 5 or len(b) < 5:
            return 1.0
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(p)

    def _min_detectable_effect(self, n: int, baseline_rate: float, alpha: float = 0.05, power: float = 0.8) -> float:
        """Approximate MDE for a proportion test."""
        if n == 0 or baseline_rate <= 0 or baseline_rate >= 1:
            # return float("inf")
            return 0.0
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        p = baseline_rate
        return (z_alpha + z_beta) * math.sqrt(2 * p * (1 - p) / n)

    def analyze(
        self,
        experiment: str,
        control_variant: str = "control",
        treatment_variant: str = "treatment",
        min_impressions: int = 100,
    ) -> ExperimentReport:

        ctrl = self._variant_metrics(experiment, control_variant)
        trt = self._variant_metrics(experiment, treatment_variant)

        comparisons: List[ComparisonResult] = []

        # ── CTR ───────────────────────────────────────────────────────────────
        p_ctr, ci_ctr = self._z_test_proportions(
            ctrl.n_impressions, ctrl.n_clicks,
            trt.n_impressions,  trt.n_clicks,
        )
        lift_ctr = (trt.ctr - ctrl.ctr) / ctrl.ctr if ctrl.ctr > 0 else 0.0
        comparisons.append(ComparisonResult(
            metric="CTR",
            control_value=ctrl.ctr, treatment_value=trt.ctr,
            relative_lift=lift_ctr, p_value=p_ctr,
            significant=p_ctr < self.alpha,
            test_used="two-proportion z-test",
            confidence_interval=ci_ctr,
        ))

        # ── Conversion rate ───────────────────────────────────────────────────
        p_conv, ci_conv = self._z_test_proportions(
            ctrl.n_impressions, ctrl.n_conversions,
            trt.n_impressions,  trt.n_conversions,
        )
        lift_conv = (trt.conversion_rate - ctrl.conversion_rate) / ctrl.conversion_rate \
            if ctrl.conversion_rate > 0 else 0.0
        comparisons.append(ComparisonResult(
            metric="Conversion rate",
            control_value=ctrl.conversion_rate, treatment_value=trt.conversion_rate,
            relative_lift=lift_conv, p_value=p_conv,
            significant=p_conv < self.alpha,
            test_used="two-proportion z-test",
            confidence_interval=ci_conv,
        ))

        # ── MRR (Mann-Whitney on rank positions) ──────────────────────────────
        p_mrr = self._mann_whitney(
            experiment, control_variant, treatment_variant, "rank", "clicks")
        lift_mrr = (trt.mrr - ctrl.mrr) / ctrl.mrr if ctrl.mrr > 0 else 0.0
        comparisons.append(ComparisonResult(
            metric="MRR",
            control_value=ctrl.mrr, treatment_value=trt.mrr,
            relative_lift=lift_mrr, p_value=p_mrr,
            significant=p_mrr < self.alpha,
            test_used="Mann-Whitney U",
        ))

        # ── Dwell time (Mann-Whitney) ─────────────────────────────────────────
        p_dwell = self._mann_whitney(
            experiment, control_variant, treatment_variant, "dwell_seconds", "clicks")
        lift_dwell = (trt.avg_dwell_seconds - ctrl.avg_dwell_seconds) / ctrl.avg_dwell_seconds \
            if ctrl.avg_dwell_seconds > 0 else 0.0
        comparisons.append(ComparisonResult(
            metric="Avg dwell (sec)",
            control_value=ctrl.avg_dwell_seconds, treatment_value=trt.avg_dwell_seconds,
            relative_lift=lift_dwell, p_value=p_dwell,
            significant=p_dwell < self.alpha,
            test_used="Mann-Whitney U",
        ))

        # ── Sample size check + MDE ────────────────────────────────────────────
        adequate = ctrl.n_impressions >= min_impressions and trt.n_impressions >= min_impressions
        mde = self._min_detectable_effect(
            min(ctrl.n_impressions, trt.n_impressions),
            ctrl.ctr or 0.1,
        )

        return ExperimentReport(
            experiment=experiment,
            control_variant=control_variant,
            treatment_variant=treatment_variant,
            variant_metrics={control_variant: ctrl, treatment_variant: trt},
            comparisons=comparisons,
            sample_size_adequate=adequate,
            min_detectable_effect=mde,
        )

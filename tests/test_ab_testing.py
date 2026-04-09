"""
tests/test_ab_testing.py

Unit tests for the A/B testing system.
Run with: pytest tests/
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.ab_testing.experiment import Experiment, Variant, ExperimentRegistry
from src.ab_testing.tracker import EventTracker, EventStore


# ── Experiment assignment ──────────────────────────────────────────────────────

class TestVariantAssignment:

    def _make_exp(self, fractions=(0.5, 0.5)):
        variants = [
            Variant(name="control",   traffic_fraction=fractions[0]),
            Variant(name="treatment", traffic_fraction=fractions[1]),
        ]
        return Experiment(name="test_exp", variants=variants, metrics=["ctr"])

    def test_assignment_is_deterministic(self):
        exp = self._make_exp()
        v1 = exp.assign("user_abc")
        v2 = exp.assign("user_abc")
        assert v1.name == v2.name

    def test_different_users_can_get_different_variants(self):
        exp = self._make_exp()
        assignments = {exp.assign(f"u{i}").name for i in range(100)}
        assert len(assignments) == 2  # both variants appear

    def test_traffic_split_is_approximately_correct(self):
        exp = self._make_exp(fractions=(0.5, 0.5))
        results = [exp.assign(f"user_{i}").name for i in range(1000)]
        ctrl_frac = results.count("control") / 1000
        # Should be ~50% ± 5%
        assert 0.45 < ctrl_frac < 0.55

    def test_invalid_fractions_raise(self):
        with pytest.raises(ValueError):
            Experiment(
                name="bad",
                variants=[
                    Variant("a", 0.3),
                    Variant("b", 0.3),
                ],
                metrics=[],
            )

    def test_80_20_split(self):
        exp = Experiment(
            name="skewed",
            variants=[Variant("ctrl", 0.8), Variant("trt", 0.2)],
            metrics=[],
        )
        results = [exp.assign(f"u{i}").name for i in range(2000)]
        ctrl_frac = results.count("ctrl") / 2000
        assert 0.75 < ctrl_frac < 0.85


# ── Event tracker ──────────────────────────────────────────────────────────────

class TestEventTracker:

    def test_writes_impression(self, tmp_path):
        log = str(tmp_path / "events.jsonl")
        with EventTracker(log) as t:
            t.log_impression("u1", "exp", "control", "python", ["d1", "d2"])
        lines = Path(log).read_text().strip().split("\n")
        assert len(lines) == 1
        import json
        ev = json.loads(lines[0])
        assert ev["event_type"] == "impression"
        assert ev["user_id"] == "u1"
        assert ev["doc_ids"] == ["d1", "d2"]

    def test_writes_click(self, tmp_path):
        log = str(tmp_path / "events.jsonl")
        with EventTracker(log) as t:
            t.log_click("u1", "exp", "treatment", "python",
                        "d2", rank=2, dwell_seconds=30.0)
        import json
        ev = json.loads(Path(log).read_text().strip())
        assert ev["event_type"] == "click"
        assert ev["rank"] == 2
        assert ev["dwell_seconds"] == 30.0

    def test_writes_conversion(self, tmp_path):
        log = str(tmp_path / "events.jsonl")
        with EventTracker(log) as t:
            t.log_conversion("u1", "exp", "treatment",
                             "python", "d2", value=5.0)
        import json
        ev = json.loads(Path(log).read_text().strip())
        assert ev["event_type"] == "conversion"
        assert ev["value"] == 5.0


# ── EventStore ─────────────────────────────────────────────────────────────────

class TestEventStore:

    def test_ingest_and_query(self, tmp_path):
        import json
        import time

        log = str(tmp_path / "events.jsonl")
        db = str(tmp_path / "events.db")

        events = [
            {"event_type": "impression", "ts": time.time(), "user_id": "u1",
             "experiment": "exp", "variant": "control", "query": "q",
             "session_id": "", "doc_ids": ["d1"], "scores": [], "page": 1},
            {"event_type": "click", "ts": time.time(), "user_id": "u1",
             "experiment": "exp", "variant": "control", "query": "q",
             "session_id": "", "doc_id": "d1", "rank": 1, "dwell_seconds": 20.0},
        ]
        with open(log, "w") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")

        store = EventStore(db)
        count = store.ingest_jsonl(log)
        assert count == 2

        imp = store.query(
            "SELECT COUNT(*) as n FROM impressions WHERE experiment='exp'")
        assert imp[0]["n"] == 1

        clk = store.query("SELECT rank FROM clicks WHERE experiment='exp'")
        assert clk[0]["rank"] == 1
        store.close()

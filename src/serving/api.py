"""
src/serving/api.py

FastAPI ranking endpoint with integrated A/B experiment assignment and event tracking.

Flow per request:
  1. Assign user to a variant (deterministic hash)
  2. Load the variant's model
  3. Rank candidate documents
  4. Log an impression event
  5. Return ranked results + experiment metadata

Additional endpoints:
  POST /event/click       — log a click event
  POST /event/conversion  — log a conversion event
  GET  /experiments       — list active experiments
  GET  /ab/report/{name}  — run statistical analysis on an experiment

Start:
    uvicorn src.serving.api:app --reload --port 8000
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.ab_testing.experiment import ExperimentRegistry
from src.ab_testing.tracker import EventStore, EventTracker
from src.data.loader import FEATURE_COLS

DEFAULT_MODEL_PATH = "outputs/model_pointwise.json"
EXPERIMENTS_YAML = "configs/experiments.yaml"
EVENTS_JSONL = "outputs/ab_events.jsonl"
EVENTS_DB = "outputs/ab_events.db"

app = FastAPI(
    title="Search Ranking API",
    description="LTR ranking with integrated A/B experiment tracking",
    version="0.2.0",
)

_model_cache: Dict[str, xgb.Booster] = {}
_tracker: Optional[EventTracker] = None
_store: Optional[EventStore] = None
_registry: Optional[ExperimentRegistry] = None


def get_tracker() -> EventTracker:
    global _tracker
    if _tracker is None:
        _tracker = EventTracker(EVENTS_JSONL)
    return _tracker


def get_store() -> EventStore:
    global _store
    if _store is None:
        _store = EventStore(EVENTS_DB)
    return _store


def get_registry() -> Optional[ExperimentRegistry]:
    global _registry
    if _registry is None and Path(EXPERIMENTS_YAML).exists():
        _registry = ExperimentRegistry.from_yaml(EXPERIMENTS_YAML)
    return _registry


def load_model(model_path: str) -> xgb.Booster:
    if model_path not in _model_cache:
        if not Path(model_path).exists():
            raise RuntimeError(
                f"Model not found: {model_path}. Run train.py first.")
        m = xgb.Booster()
        m.load_model(model_path)
        _model_cache[model_path] = m
    return _model_cache[model_path]


class Candidate(BaseModel):
    doc_id: str
    bm25_score: float = 0.0
    tfidf_cosine: float = 0.0
    query_term_coverage: float = 0.0
    title_match_score: float = 0.0
    body_match_score: float = 0.0
    doc_length: float = 500.0
    doc_pagerank: float = 0.1
    doc_freshness_days: float = 100.0
    avg_click_rate: float = 0.0
    query_length: float = 3.0
    query_idf_sum: float = 10.0
    is_navigational: float = 0.0


class RankRequest(BaseModel):
    query: str
    candidates: List[Candidate]
    user_id: str = "anonymous"
    session_id: str = ""
    experiment: Optional[str] = None


class RankedDocument(BaseModel):
    doc_id: str
    rank: int
    score: float


class RankResponse(BaseModel):
    query: str
    results: List[RankedDocument]
    experiment: Optional[str] = None
    variant: Optional[str] = None


class ClickRequest(BaseModel):
    user_id: str
    experiment: str
    variant: str
    query: str
    doc_id: str
    rank: int
    session_id: str = ""
    dwell_seconds: Optional[float] = None


class ConversionRequest(BaseModel):
    user_id: str
    experiment: str
    variant: str
    query: str
    doc_id: str
    session_id: str = ""
    conversion_type: str = "default"
    value: float = 1.0


def rank_candidates(model: xgb.Booster, candidates: List[Candidate]):
    X = np.array(
        [[getattr(c, f) for f in FEATURE_COLS] for c in candidates],
        dtype=np.float32
    )
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    dmatrix = xgb.DMatrix(X, feature_names=FEATURE_COLS)
    dmatrix.set_group([len(candidates)])
    scores = model.predict(dmatrix)
    X = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    order = np.argsort(-scores)
    return order.tolist(), scores.tolist()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/rank", response_model=RankResponse)
def rank(request: RankRequest):
    if not request.candidates:
        raise HTTPException(status_code=400, detail="candidates list is empty")

    experiment_name = request.experiment or "exp_ltr_v2"
    assigned_variant = None
    model_path = DEFAULT_MODEL_PATH

    if experiment_name:
        registry = get_registry()
        if registry is None:
            raise HTTPException(
                status_code=503, detail="Experiment registry not loaded")
        try:
            variant = registry.assign(experiment_name, request.user_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

        if variant is None:
            experiment_name = None
        else:
            assigned_variant = variant.name
            if variant.model_path:
                model_path = variant.model_path

    model = load_model(model_path)
    order, scores = rank_candidates(model, request.candidates)

    ranked_docs = [
        RankedDocument(
            doc_id=request.candidates[i].doc_id, rank=pos + 1, score=float(scores[i]))
        for pos, i in enumerate(order)
    ]

    if experiment_name and assigned_variant:
        get_tracker().log_impression(
            user_id=request.user_id, experiment=experiment_name,
            variant=assigned_variant, query=request.query,
            session_id=request.session_id,
            doc_ids=[d.doc_id for d in ranked_docs],
            scores=[d.score for d in ranked_docs],
        )

    return RankResponse(query=request.query, results=ranked_docs,
                        experiment=experiment_name, variant=assigned_variant)


@app.post("/event/click")
def log_click(req: ClickRequest):
    get_tracker().log_click(
        user_id=req.user_id, experiment=req.experiment, variant=req.variant,
        query=req.query, doc_id=req.doc_id, rank=req.rank,
        session_id=req.session_id, dwell_seconds=req.dwell_seconds,
    )
    return {"status": "ok"}


@app.post("/event/conversion")
def log_conversion(req: ConversionRequest):
    get_tracker().log_conversion(
        user_id=req.user_id, experiment=req.experiment, variant=req.variant,
        query=req.query, doc_id=req.doc_id, session_id=req.session_id,
        conversion_type=req.conversion_type, value=req.value,
    )
    return {"status": "ok"}


@app.get("/experiments")
def list_experiments():
    registry = get_registry()
    if registry is None:
        return {"active": []}
    return {"active": registry.list_active()}


def safe(v):
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return 0.0
    return v

@app.get("/ab/report/{experiment_name}")
def ab_report(experiment_name: str, control: str = "control", treatment: str = "treatment"):
    from src.ab_testing.analysis import ExperimentAnalyzer

    store = get_store()
    if Path(EVENTS_JSONL).exists():
        store.ingest_jsonl(EVENTS_JSONL)

    analyzer = ExperimentAnalyzer(store)
    try:
        report = analyzer.analyze(experiment_name, control, treatment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "experiment": report.experiment,
        "sample_size_adequate": report.sample_size_adequate,
        "min_detectable_effect": safe(report.min_detectable_effect),
        "variants": {
            name: {
                "n_impressions": vm.n_impressions,
                "n_clicks": vm.n_clicks,
                "ctr": safe(vm.ctr),
                "n_conversions": vm.n_conversions,
                "conversion_rate": safe(vm.conversion_rate),
                "mrr": safe(vm.mrr),
                "avg_rank_clicked": safe(vm.avg_rank_clicked),
            }
            for name, vm in report.variant_metrics.items()
        },
        "comparisons": [
            {
                "metric": c.metric,
                "control": safe(c.control_value),
                "treatment": safe(c.treatment_value),
                "relative_lift": safe(c.relative_lift),
                "p_value": safe(c.p_value),
                "significant": c.significant,
                "test": c.test_used,
            }
            for c in report.comparisons
        ],
    }

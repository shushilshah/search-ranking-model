"""
src/ab_testing/tracker.py

Online event tracker for A/B experiments.

Tracks three event types:
  - impression  : a ranked list was shown to the user
  - click       : user clicked a result at a given rank position
  - conversion  : user completed a goal action (purchase, download, etc.)

Events are written as newline-delimited JSON to a log file, which can be
shipped to any data warehouse. The EventStore can also be backed by SQLite
for lightweight local analytics without a warehouse.

Usage:
    tracker = EventTracker(log_path="outputs/ab_events.jsonl")
    tracker.log_impression(user_id="u1", experiment="exp_ltr_v2", variant="treatment",
                           query="python tutorial", doc_ids=["d1","d2","d3"])
    tracker.log_click(user_id="u1", experiment="exp_ltr_v2", variant="treatment",
                      query="python tutorial", doc_id="d2", rank=2)
"""

from __future__ import annotations
import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


# ── Event models ───────────────────────────────────────────────────────────────

@dataclass
class ImpressionEvent:
    event_type: str = "impression"
    ts: float = field(default_factory=time.time)
    user_id: str = ""
    experiment: str = ""
    variant: str = ""
    query: str = ""
    session_id: str = ""
    doc_ids: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    page: int = 1


@dataclass
class ClickEvent:
    event_type: str = "click"
    ts: float = field(default_factory=time.time)
    user_id: str = ""
    experiment: str = ""
    variant: str = ""
    query: str = ""
    session_id: str = ""
    doc_id: str = ""
    rank: int = 0         # 1-based rank position clicked
    dwell_seconds: Optional[float] = None


@dataclass
class ConversionEvent:
    event_type: str = "conversion"
    ts: float = field(default_factory=time.time)
    user_id: str = ""
    experiment: str = ""
    variant: str = ""
    query: str = ""
    session_id: str = ""
    doc_id: str = ""
    conversion_type: str = "default"   # e.g. "purchase", "download", "signup"
    value: float = 1.0                 # optional monetary / score value


# ── JSON-lines tracker ─────────────────────────────────────────────────────────

class EventTracker:
    """
    Writes events as newline-delimited JSON to a log file.
    Thread-safe for single-process use; for multi-process append the file
    atomically or use EventStore (SQLite-backed) instead.
    """

    def __init__(self, log_path: str = "outputs/ab_events.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.log_path.open("a", buffering=1)  # line-buffered

    def _write(self, event) -> None:
        self._fh.write(json.dumps(asdict(event)) + "\n")

    def log_impression(
        self,
        user_id: str,
        experiment: str,
        variant: str,
        query: str,
        doc_ids: List[str],
        scores: Optional[List[float]] = None,
        session_id: str = "",
        page: int = 1,
    ) -> ImpressionEvent:
        ev = ImpressionEvent(
            user_id=user_id, experiment=experiment, variant=variant,
            query=query, session_id=session_id, doc_ids=doc_ids,
            scores=scores or [], page=page,
        )
        self._write(ev)
        return ev

    def log_click(
        self,
        user_id: str,
        experiment: str,
        variant: str,
        query: str,
        doc_id: str,
        rank: int,
        session_id: str = "",
        dwell_seconds: Optional[float] = None,
    ) -> ClickEvent:
        ev = ClickEvent(
            user_id=user_id, experiment=experiment, variant=variant,
            query=query, session_id=session_id, doc_id=doc_id,
            rank=rank, dwell_seconds=dwell_seconds,
        )
        self._write(ev)
        return ev

    def log_conversion(
        self,
        user_id: str,
        experiment: str,
        variant: str,
        query: str,
        doc_id: str,
        session_id: str = "",
        conversion_type: str = "default",
        value: float = 1.0,
    ) -> ConversionEvent:
        ev = ConversionEvent(
            user_id=user_id, experiment=experiment, variant=variant,
            query=query, session_id=session_id, doc_id=doc_id,
            conversion_type=conversion_type, value=value,
        )
        self._write(ev)
        return ev

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── SQLite-backed event store (optional, for local analytics) ─────────────────

class EventStore:
    """
    SQLite store for lightweight local A/B analytics.
    Supports bulk load from a JSONL log file.
    """

    def __init__(self, db_path: str = "outputs/ab_events.db"):
        self.db_path = db_path
        self._con = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        self._con.executescript("""
            CREATE TABLE IF NOT EXISTS impressions (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        REAL,
                user_id   TEXT,
                experiment TEXT,
                variant   TEXT,
                query     TEXT,
                session_id TEXT,
                n_docs    INTEGER,
                page      INTEGER
            );
            CREATE TABLE IF NOT EXISTS clicks (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        REAL,
                user_id   TEXT,
                experiment TEXT,
                variant   TEXT,
                query     TEXT,
                session_id TEXT,
                doc_id    TEXT,
                rank      INTEGER,
                dwell_seconds REAL
            );
            CREATE TABLE IF NOT EXISTS conversions (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        REAL,
                user_id   TEXT,
                experiment TEXT,
                variant   TEXT,
                query     TEXT,
                session_id TEXT,
                doc_id    TEXT,
                conversion_type TEXT,
                value     REAL
            );
            CREATE INDEX IF NOT EXISTS idx_imp_exp  ON impressions(experiment, variant);
            CREATE INDEX IF NOT EXISTS idx_clk_exp  ON clicks(experiment, variant);
            CREATE INDEX IF NOT EXISTS idx_conv_exp ON conversions(experiment, variant);
        """)
        self._con.commit()

    def ingest_jsonl(self, path: str) -> int:
        """Load events from a JSONL log file into SQLite. Returns rows ingested."""
        count = 0
        with open(path) as f:
            for line in f:
                ev = json.loads(line.strip())
                etype = ev.get("event_type")
                if etype == "impression":
                    self._con.execute(
                        "INSERT INTO impressions (ts,user_id,experiment,variant,query,session_id,n_docs,page) "
                        "VALUES (?,?,?,?,?,?,?,?)",
                        (ev["ts"], ev["user_id"], ev["experiment"], ev["variant"],
                         ev["query"], ev["session_id"], len(ev.get("doc_ids", [])), ev.get("page", 1))
                    )
                elif etype == "click":
                    self._con.execute(
                        "INSERT INTO clicks (ts,user_id,experiment,variant,query,session_id,doc_id,rank,dwell_seconds) "
                        "VALUES (?,?,?,?,?,?,?,?,?)",
                        (ev["ts"], ev["user_id"], ev["experiment"], ev["variant"],
                         ev["query"], ev["session_id"], ev["doc_id"], ev["rank"], ev.get("dwell_seconds"))
                    )
                elif etype == "conversion":
                    self._con.execute(
                        "INSERT INTO conversions (ts,user_id,experiment,variant,query,session_id,doc_id,conversion_type,value) "
                        "VALUES (?,?,?,?,?,?,?,?,?)",
                        (ev["ts"], ev["user_id"], ev["experiment"], ev["variant"],
                         ev["query"], ev["session_id"], ev["doc_id"], ev["conversion_type"], ev["value"])
                    )
                count += 1
        self._con.commit()
        return count

    def query(self, sql: str, params=()) -> list:
        cur = self._con.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self):
        self._con.close()

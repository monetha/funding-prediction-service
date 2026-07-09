"""Append-only sqlite log of every production prediction (the ones served with save=true):
metadata + the 65-feature vector + p_sl. Later joined to funding_bot.db (symbol + nearest
opened_at ~ event_time) to attach labels for retraining, and to audit serve/train feature
drift. Callers that do not set save=true never reach this module.

Contract: logging a prediction must NEVER fail the request. `log()` swallows and reports
its own errors and returns the row id or None.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DB = Path(os.environ.get(
    "SLP_FEATURE_STORE_DB",
    str(Path(__file__).resolve().parent / "feature_store.db")))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    requested_at  TEXT NOT NULL,     -- ISO8601 UTC, when the request was served
    event_time    TEXT NOT NULL,     -- the sampled bar (feature timestamp)
    symbol        TEXT NOT NULL,     -- CCXT unified symbol as requested
    exchange      TEXT NOT NULL,
    model_version TEXT,
    p_sl          REAL,
    features_json TEXT NOT NULL      -- the 65-feature vector as JSON {name: value}
);
CREATE INDEX IF NOT EXISTS ix_pred_symbol_event ON prediction_log (symbol, event_time);
"""


class FeatureStore:
    """Thread-safe append log. sqlite inserts are sub-ms; a single guarded connection
    (check_same_thread=False + lock) is enough for the serving write rate."""

    def __init__(self, db_path: Path | str = DEFAULT_DB):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._con = sqlite3.connect(self.db_path, check_same_thread=False)
        self._con.executescript(_SCHEMA)
        self._con.commit()

    def log(self, *, requested_at: str, event_time: str, symbol: str, exchange: str,
            model_version, p_sl: float, features: dict) -> int | None:
        """Append one prediction row. Returns row id, or None on failure (never raises)."""
        try:
            with self._lock:
                cur = self._con.execute(
                    "INSERT INTO prediction_log (requested_at, event_time, symbol, "
                    "exchange, model_version, p_sl, features_json) VALUES (?,?,?,?,?,?,?)",
                    (requested_at, event_time, symbol, exchange,
                     str(model_version) if model_version is not None else None,
                     float(p_sl),
                     json.dumps({k: _finite(v) for k, v in features.items()})),
                )
                self._con.commit()
                return cur.lastrowid
        except Exception as exc:                     # logging must never fail a response
            logger.warning("feature_store.log failed (non-fatal): %s", exc)
            return None

    def count(self) -> int:
        with self._lock:
            return self._con.execute("SELECT COUNT(*) FROM prediction_log").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._con.close()


def _finite(v):
    """JSON can't encode NaN/inf; store them as None so the row is still valid."""
    try:
        f = float(v)
        return f if f == f and f not in (float("inf"), float("-inf")) else None
    except (TypeError, ValueError):
        return v

"""Train/serve feature parity — the plan's #1 risk.

Because featuregen is CAUSAL (features at bar t use only bars <= t), sampling at a
fixed past event_time yields identical features no matter how many newer bars a frame
has. So an offline recompute at the service's returned event_time must reproduce the
service's p_sl exactly (same code, same pinned env).

Two tests:
  * test_offline_determinism — no service needed: the same klines through the SAME code
    path (generate -> sample_at -> production model) twice, must match to ~1e-9, be NaN
    free, and use exactly the feats.json column order.
  * test_service_matches_offline — hits a running service (skipped if unavailable) and
    asserts service p_sl == offline p_sl at the same bar to ~1e-6.

    pytest tests/test_parity.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from featuregen import FeatureGenerator            # noqa: E402
from serving.binance_klines import fetch_raw_1m_sync  # noqa: E402

MLFLOW_URI = os.environ.get("SLP_MLFLOW_URI", "http://127.0.0.1:5000")
SERVICE_URI = os.environ.get("SLP_SERVICE_URI", "http://127.0.0.1:8100")
SYMBOL = os.environ.get("SLP_PARITY_SYMBOL", "SOL/USDT:USDT")
TOL = 1e-6


def _load_production():
    """(model, feats) from the tracking server, or skip if unreachable."""
    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        client = MlflowClient()
        mv = client.get_model_version_by_alias("sl_classifier", "production")
        model = mlflow.sklearn.load_model("models:/sl_classifier@production")
        import json
        fp = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="feats.json")
        with open(fp) as fh:
            feats = json.load(fh)
        return model, feats, mv.version
    except Exception as exc:
        pytest.skip(f"MLflow production model unavailable: {exc}")


def _offline_p_sl(model, feats, symbol, sample_ts):
    """Reproduce the service computation offline for one bar."""
    df1m = fetch_raw_1m_sync(symbol, limit=300)
    btc = fetch_raw_1m_sync("BTCUSDT", limit=300)
    fg = FeatureGenerator(btc=btc)
    frame = fg.generate(df1m)
    row = fg.sample_at(frame, [sample_ts])
    X = row.reindex(columns=feats).astype("float32")
    return float(model.predict_proba(X)[:, 1][0]), X


def test_offline_determinism():
    model, feats, _ = _load_production()
    assert len(feats) == 65
    # a fixed, safely-in-range past bar
    sample_ts = pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=2)
    p1, X1 = _offline_p_sl(model, feats, SYMBOL, sample_ts)
    p2, X2 = _offline_p_sl(model, feats, SYMBOL, sample_ts)
    assert list(X1.columns) == feats                      # exact training column order
    assert not X1.isna().any(axis=None), "NaN features on a warm symbol"
    assert 0.0 <= p1 <= 1.0
    assert abs(p1 - p2) < 1e-9                             # deterministic


def test_service_matches_offline():
    try:
        r = requests.post(f"{SERVICE_URI}/predict",
                          json={"symbol": SYMBOL, "exchange": "binance"}, timeout=10)
    except requests.RequestException as exc:
        pytest.skip(f"service not running: {exc}")
    if r.status_code != 200:
        pytest.skip(f"service returned {r.status_code}: {r.text}")
    body = r.json()
    p_service = body["p_sl"]
    event_time = pd.Timestamp(body["event_time"])         # the bar the service sampled

    model, feats, version = _load_production()
    assert str(version) == str(body["model_version"])
    p_offline, _ = _offline_p_sl(model, feats, SYMBOL, event_time)
    assert abs(p_service - p_offline) < TOL, (
        f"parity break: service={p_service:.8f} offline={p_offline:.8f} "
        f"diff={abs(p_service - p_offline):.2e} @ {event_time}")

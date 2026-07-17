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
from serving.binance_klines import fetch_raw_1m_sync, parse_klines  # noqa: E402

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


def _bar(ts: pd.Timestamp) -> list:
    """A minimal 12-field Binance kline whose only meaningful field is the open time."""
    return [int(ts.timestamp() * 1000)] + [1.0] * 11


def test_partial_bar_dropped_only_when_really_partial():
    """Regression: production served ~1/3 of illiquid-token vectors from the WRONG bar.

    Binance creates a bar only once a trade occurs, so early in a new minute an illiquid
    symbol's last element is already CLOSED. parse_klines used to drop the last element
    unconditionally, discarding that newest closed bar; sample_at's ffill then silently
    returned the bar BEFORE the requested one (experiments/FINDINGS.md). No network: the
    two cases are constructed directly.
    """
    now_min = pd.Timestamp.now(tz="UTC").floor("min")
    sample_ts = now_min - pd.Timedelta(minutes=1)

    # liquid: the current minute has a (still-forming) bar -> it MUST be dropped
    liquid = parse_klines([_bar(now_min - pd.Timedelta(minutes=k)) for k in (2, 1, 0)])
    # illiquid: no bar for the current minute yet -> there is nothing to drop
    illiquid = parse_klines([_bar(now_min - pd.Timedelta(minutes=k)) for k in (3, 2, 1)])

    assert liquid.index[-1] == sample_ts, "in-progress bar was not dropped"
    assert illiquid.index[-1] == sample_ts, (
        "the newest CLOSED bar was dropped — features would be served from the "
        "previous minute under the requested event_time")


def test_btc_frame_must_reach_the_sample_bar():
    """Regression: every served btc_* feature was computed from a 1-minute-stale BTC frame.

    `_btc_block` reindex-ffills BTC onto the token index, so a frame one bar short does not
    error — it silently freezes btc_* at the previous bar, which is what production did on
    16/16 rows. The old wall-clock guard (BTC_MAX_AGE_SEC=90) could never catch that, since
    1-bar staleness is always within 90s. `_btc_for` must test bar ALIGNMENT instead.
    No network: the refetch is stubbed.
    """
    import asyncio

    from serving import app as app_mod

    sample_ts = pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=1)

    def _frame(last: pd.Timestamp) -> pd.DataFrame:
        idx = pd.date_range(end=last, periods=300, freq="1min", tz="UTC")
        return pd.DataFrame({"close": 1.0, "tbb": 1.0, "volume": 1.0}, index=idx)

    stale = _frame(sample_ts - pd.Timedelta(minutes=1))   # one bar short — the real bug
    fresh = _frame(sample_ts)

    app_mod.state.btc = stale
    app_mod.state.btc_fetched_at = __import__("time").monotonic()   # young but MISALIGNED

    refreshed = []

    async def _fake_refresh():
        refreshed.append(True)
        app_mod.state.btc = fresh                        # the inline refetch reaches the bar
        return True

    orig = app_mod._refresh_btc_once
    app_mod._refresh_btc_once = _fake_refresh
    try:
        got = asyncio.run(app_mod._btc_for(sample_ts))
        assert refreshed, "a misaligned BTC frame was accepted without refetching"
        assert got is not None and got.index[-1] >= sample_ts

        # already-aligned frame must be used as-is, with no refetch
        refreshed.clear()
        app_mod.state.btc = fresh
        got = asyncio.run(app_mod._btc_for(sample_ts))
        assert got is not None and not refreshed, "refetched despite an aligned frame"

        # refetch that still cannot reach the sample bar -> fail closed, never ffill
        async def _fake_bad_refresh():
            app_mod.state.btc = stale
            return False

        app_mod._refresh_btc_once = _fake_bad_refresh
        app_mod.state.btc = stale
        assert asyncio.run(app_mod._btc_for(sample_ts)) is None, "did not fail closed"
    finally:
        app_mod._refresh_btc_once = orig
        app_mod.state.btc = None


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

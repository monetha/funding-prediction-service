"""SL-Probability Prediction Service — FastAPI.

Request-only: the funding bot sends {symbol, exchange, save} and gets back a raw p_sl; the
bot owns all sizing. Binance-trained -> non-Binance requests are rejected (off-distribution).

Design (see plan lively-spinning-moon.md):
  * startup loads the @production model + its ordered feats.json ONCE into memory;
    requests never touch MLflow. /reload hot-swaps after a retrain.
  * a background refresher fetches the shared BTC frame every ~25s into memory, so the hot
    path normally does ZERO BTC fetches; a request refetches inline (once, under a lock)
    only when the cached frame does not reach its sample bar. Token klines get a short
    per-symbol TTL to absorb burst duplicates.
  * a prediction sent with save=true appends its 65-feature vector + p_sl to the sqlite
    feature store (non-fatal) for later labeling / retraining. save defaults to false, so
    only the calls the bot marks as real production predictions enter the training log.
  * fail-closed: BTC that cannot reach the sample bar, no token bar AT the sample bar, too
    little token history, or any NaN feature -> error, never a silent fall back to the
    57-feature (no-BTC) model input and never a vector quietly sampled at the wrong bar.

    uvicorn serving.app:app --host 0.0.0.0 --port 8100
"""
from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

from featuregen import FeatureGenerator
from serving.binance_klines import fetch_raw_1m
from serving.feature_store import FeatureStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRACKING_URI = os.environ.get("SLP_MLFLOW_URI", "http://127.0.0.1:5000")
REGISTERED_MODEL = "sl_classifier"
PROD_ALIAS = "production"

BTC_SYMBOL = "BTCUSDT"
BTC_REFRESH_SEC = float(os.environ.get("SLP_BTC_REFRESH_SEC", "25"))
BTC_MAX_AGE_SEC = float(os.environ.get("SLP_BTC_MAX_AGE_SEC", "90"))   # fail-closed above
TOKEN_TTL_SEC = float(os.environ.get("SLP_TOKEN_TTL_SEC", "8"))
KLINE_LIMIT = 300
MIN_BARS = 241                    # need >=241 usable bars for the 240-window features

# Every /predict call — served or rejected, save=true or not — gets one JSON line here.
# This is the request audit trail; the feature store only holds save=true rows.
LOG_DIR = os.environ.get("SLP_LOG_DIR")
LOG_MAX_BYTES = int(os.environ.get("SLP_LOG_MAX_BYTES", str(50 * 1024 * 1024)))
LOG_BACKUPS = int(os.environ.get("SLP_LOG_BACKUPS", "5"))

predict_log = logging.getLogger("predictions")
predict_log.propagate = False     # keep JSON lines out of the uvicorn stdout stream


def _init_predict_log() -> None:
    """Attach a rotating predictions.log handler when SLP_LOG_DIR is set. Never fatal:
    an unwritable log directory must not stop the service from serving."""
    if not LOG_DIR or predict_log.handlers:
        return
    try:
        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
        h = logging.handlers.RotatingFileHandler(
            Path(LOG_DIR) / "predictions.log",
            maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS)
        h.setFormatter(logging.Formatter("%(message)s"))   # the record IS the JSON line
        predict_log.addHandler(h)
        predict_log.setLevel(logging.INFO)
        logger.info("prediction log -> %s", Path(LOG_DIR) / "predictions.log")
    except Exception as exc:
        logger.warning("prediction log disabled (non-fatal): %s", exc)


def _log_predict(**fields) -> None:
    """One JSON line per /predict call. Never raises."""
    if not predict_log.handlers:
        return
    try:
        predict_log.info(json.dumps(
            {"ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **fields}))
    except Exception as exc:
        logger.warning("prediction log write failed (non-fatal): %s", exc)


class State:
    """In-memory service state (single process). Attribute assignment is atomic under
    the GIL, so the refresher writing `btc` while a request reads it is safe."""
    def __init__(self):
        self.model = None
        self.feats: list[str] = []
        self.model_version = None
        self.run_id = None
        self.btc: pd.DataFrame | None = None
        self.btc_fetched_at: float = 0.0        # time.monotonic() of last BTC fetch
        self.session: aiohttp.ClientSession | None = None
        self.store: FeatureStore | None = None
        self.token_cache: dict[str, tuple[float, pd.DataFrame]] = {}

    def btc_age(self) -> float | None:
        return None if self.btc is None else time.monotonic() - self.btc_fetched_at


state = State()


class PredictRequest(BaseModel):
    symbol: str
    exchange: str
    # only a real production prediction is persisted for retraining; probes, tests and
    # manual curls leave the training log untouched.
    save: bool = False


# --------------------------------------------------------------------------- #
# model loading (startup + /reload)
# --------------------------------------------------------------------------- #
def load_production() -> tuple:
    """Load the @production model + its ordered feats.json + version. Returns
    (model, feats, version, run_id)."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL, PROD_ALIAS)
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL}@{PROD_ALIAS}")
    feats_path = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="feats.json")
    with open(feats_path) as fh:
        feats = json.load(fh)
    logger.info("loaded %s v%s (%d feats) run=%s",
                REGISTERED_MODEL, mv.version, len(feats), mv.run_id[:12])
    return model, feats, mv.version, mv.run_id


# --------------------------------------------------------------------------- #
# BTC background refresher
# --------------------------------------------------------------------------- #
async def _refresh_btc_once() -> bool:
    try:
        btc = await fetch_raw_1m(state.session, BTC_SYMBOL, limit=KLINE_LIMIT)
        if len(btc) < MIN_BARS:
            logger.warning("BTC refresh returned only %d bars", len(btc))
            return False
        state.btc = btc
        state.btc_fetched_at = time.monotonic()
        return True
    except Exception as exc:
        logger.warning("BTC refresh failed: %s", exc)
        return False


async def _btc_refresher():
    while True:
        await _refresh_btc_once()
        await asyncio.sleep(BTC_REFRESH_SEC)


# Serialises the on-demand refresh below: the bot fires several symbols at once just after
# the hour, and without this every one of them would launch its own BTC fetch.
_btc_lock = asyncio.Lock()


async def _btc_for(sample_ts: pd.Timestamp) -> pd.DataFrame | None:
    """The BTC frame to build features with, guaranteed to REACH `sample_ts`, or None.

    Wall-clock age is the wrong correctness test. `_btc_block` reindex-ffills BTC onto the
    token index, so a frame whose last bar is one minute short does not fail — it silently
    freezes every btc_* feature at the previous bar. The background refresher is free-running
    (BTC_REFRESH_SEC), and requests land a few seconds after the minute boundary, so its
    newest closed bar was reliably one minute behind the sample bar: measured drift on 16/16
    production rows (experiments/FINDINGS.md). BTC_MAX_AGE_SEC=90 never caught it because
    1-bar staleness is always well within it.

    So test bar alignment, not age, and refetch inline when the cached frame is short. BTC is
    liquid enough that its sample-minute bar always exists, making an inline fetch reliable;
    fail closed if even that comes up short.

    Cost: at the bot's cadence the refresher is usually 1 bar behind, so the inline fetch is
    the COMMON path at a minute boundary, not a rare one — the old "zero BTC fetches on the
    hot path" property is gone. The lock keeps it to one fetch per boundary (~250ms for the
    first caller, which the rest then reuse) rather than one per symbol.
    """
    btc = state.btc
    if btc is not None and len(btc) and btc.index[-1] >= sample_ts:
        return btc
    async with _btc_lock:
        btc = state.btc                    # another request may have refreshed us while we waited
        if btc is not None and len(btc) and btc.index[-1] >= sample_ts:
            return btc
        await _refresh_btc_once()
        btc = state.btc
    if btc is None or not len(btc) or btc.index[-1] < sample_ts:
        return None
    return btc


# --------------------------------------------------------------------------- #
# lifespan: load model, open session/store, start refresher
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_predict_log()
    state.model, state.feats, state.model_version, state.run_id = load_production()
    state.session = aiohttp.ClientSession()
    state.store = FeatureStore()
    await _refresh_btc_once()                      # warm BTC before serving
    task = asyncio.create_task(_btc_refresher())
    logger.info("service ready (BTC age %.1fs)", state.btc_age() or -1)
    try:
        yield
    finally:
        task.cancel()
        if state.session:
            await state.session.close()
        if state.store:
            state.store.close()


app = FastAPI(title="SL-Probability Prediction Service", lifespan=lifespan)


async def _get_token_klines(symbol: str, sample_ts: pd.Timestamp) -> pd.DataFrame:
    """Per-symbol TTL-cached token klines (dedupes burst/retry). Never fetches BTC.

    A frame is only reusable if it actually reaches `sample_ts`: the TTL can span a minute
    boundary, and a frame fetched last minute would otherwise be served for this minute's
    bar and ffilled back to a stale row."""
    now = time.monotonic()
    cached = state.token_cache.get(symbol)
    if cached and now - cached[0] < TOKEN_TTL_SEC:
        df = cached[1]
        if len(df) and df.index[-1] >= sample_ts:
            return df
    df = await fetch_raw_1m(state.session, symbol, limit=KLINE_LIMIT)
    state.token_cache[symbol] = (now, df)
    return df


@app.post("/predict")
async def predict(req: PredictRequest):
    started = time.monotonic()
    try:
        return await _predict(req, started)
    except HTTPException as exc:                  # log the reject, then fail as before
        _log_predict(symbol=req.symbol, exchange=req.exchange, save=req.save,
                     status=exc.status_code, detail=exc.detail,
                     latency_ms=round((time.monotonic() - started) * 1000, 1))
        raise


async def _predict(req: PredictRequest, started: float):
    if req.exchange.lower() != "binance":
        raise HTTPException(status_code=400, detail="unsupported exchange")
    if state.model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    # Fix sample_ts BEFORE fetching: pinning the target bar first means the frames are
    # validated against it, rather than the bar being inferred from whatever arrived.
    sample_ts = pd.Timestamp.now(tz=timezone.utc).floor("min") - pd.Timedelta(minutes=1)

    # fail-closed on a BTC frame that cannot reach the sample bar — never drop to the
    # 57-feature input, and never let the reindex-ffill quietly stale the btc_* features
    btc = await _btc_for(sample_ts)
    if btc is None:
        raise HTTPException(status_code=503, detail="btc frame unavailable/stale")

    try:
        df1m = await _get_token_klines(req.symbol, sample_ts)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"kline fetch failed: {exc}")
    if len(df1m) < MIN_BARS:
        raise HTTPException(status_code=422, detail="insufficient history")

    fg = FeatureGenerator(btc=btc)
    frame = fg.generate(df1m)
    # sample_at ffills, so a missing sample bar silently yields an EARLIER bar's features
    # logged under the requested event_time. dataset_csv.fill_from_rest already refuses to
    # write such a row; serving must refuse to serve one.
    if sample_ts not in frame.index:
        raise HTTPException(status_code=422, detail="no kline bar at the sample timestamp")
    row = fg.sample_at(frame, [sample_ts])
    X = row.reindex(columns=state.feats).astype("float32")
    if bool(X.isna().any(axis=None)):
        raise HTTPException(status_code=422, detail="insufficient history")

    p_sl = float(state.model.predict_proba(X)[:, 1][0])
    event_time = sample_ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    # persist the feature vector for future labeling/retraining (never fatal)
    saved = False
    if req.save:
        row_id = state.store.log(
            requested_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            event_time=event_time, symbol=req.symbol, exchange=req.exchange,
            model_version=state.model_version, p_sl=p_sl,
            features={k: X.iloc[0][k] for k in state.feats},
        )
        saved = row_id is not None
    _log_predict(symbol=req.symbol, exchange=req.exchange, save=req.save, status=200,
                 p_sl=round(p_sl, 6), model_version=state.model_version,
                 event_time=event_time, saved=saved,
                 latency_ms=round((time.monotonic() - started) * 1000, 1))
    return {"p_sl": p_sl, "model_version": state.model_version,
            "event_time": event_time, "saved": saved}


@app.get("/health")
async def health():
    age = state.btc_age()
    return {
        "status": "ok" if (state.model is not None and age is not None
                           and age <= BTC_MAX_AGE_SEC) else "degraded",
        "model_loaded": state.model is not None,
        "model_version": state.model_version,
        "n_features": len(state.feats),
        "btc_age_sec": None if age is None else round(age, 1),
        "predictions_logged": state.store.count() if state.store else None,
    }


@app.post("/reload")
async def reload():
    try:
        state.model, state.feats, state.model_version, state.run_id = load_production()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"reload failed: {exc}")
    return {"reloaded": True, "model_version": state.model_version, "n_features": len(state.feats)}

"""Monthly retrain — schedule-agnostic, idempotent CLI. A scheduler (cron) invokes this;
MLflow is a step inside, it does not schedule anything.

Steps:
  1. DATASET      dataset_csv.sync() -> the training set. It owns ALL feature sourcing:
                  cached dataset.csv, then feature_store.db (served vectors), then Binance
                  REST for anything still missing. Fails fast if featuregen.py changed.
  2. RETRAIN      train.train(dataset) -> log to MLflow, AUC gate, promote @production.
  3. RELOAD       POST /reload on the service (non-fatal).
  4. DRIFT AUDIT  for matched bars, diff logged features vs features recomputed from the
                  cache -> serve/train drift report. Audit only, never a training source.

    python retrain_monthly.py                 # full run
    python retrain_monthly.py --skip-reload   # e.g. offline retrain
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3

import pandas as pd
import requests

import dataset_csv
import train as train_mod
from data_sources import DEFAULT_FUNDING_DB, KlineCache, load_closed_positions
from dataset import FEATURE_LAG_MIN, attach_target
from featuregen import FeatureGenerator
from model import META_COLS, feature_cols
from serving.feature_store import DEFAULT_DB as FEATURE_STORE_DB

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SERVICE_URI = os.environ.get("SLP_SERVICE_URI", "http://127.0.0.1:8100")
MATCH_TOL = pd.Timedelta("5min")     # event_time ~ opened_at - 1min; allow slack


def _read_prediction_log(db_path=FEATURE_STORE_DB) -> pd.DataFrame:
    """Served predictions with their logged feature vectors expanded to columns."""
    if not os.path.exists(db_path):
        return pd.DataFrame()
    con = sqlite3.connect(str(db_path))
    preds = pd.read_sql_query(
        "SELECT symbol, event_time, model_version, p_sl, features_json FROM prediction_log", con)
    con.close()
    if preds.empty:
        return preds
    preds["token"] = preds["symbol"].str.split("/").str[0]
    # event_time is ISO UTC ('...Z'); make tz-naive UTC to match funding_bot opened_at
    preds["event_ts"] = pd.to_datetime(preds["event_time"], utc=True).dt.tz_localize(None)
    feats = pd.json_normalize(preds["features_json"].map(json.loads))
    return pd.concat([preds.drop(columns=["features_json"]), feats], axis=1)


def label_feature_log(funding_db=DEFAULT_FUNDING_DB, store_db=FEATURE_STORE_DB) -> pd.DataFrame:
    """Match served predictions to closed positions and attach META+target, keeping the
    SERVED feature columns. Returns a dataset-shaped frame (META_COLS + 65 feats)."""
    preds = _read_prediction_log(store_db)
    if preds.empty:
        logger.info("no served predictions to label")
        return pd.DataFrame()

    positions = attach_target(load_closed_positions(funding_db))
    feat_names = [c for c in preds.columns
                  if c not in ("symbol", "event_time", "event_ts", "token",
                               "model_version", "p_sl")]

    preds = preds.sort_values("event_ts")
    pos = positions.sort_values("opened_at")
    matched = pd.merge_asof(
        preds, pos[["position_id", "token", "opened_at", "closed_at", "symbol",
                    "funding_period", "funding_rate", "entry_price", "intended_close",
                    "margin_min", "realized_pnl", "target"]],
        left_on="event_ts", right_on="opened_at", by="token",
        direction="nearest", tolerance=MATCH_TOL, suffixes=("", "_pos"))
    matched = matched.dropna(subset=["position_id"])
    logger.info("labeled %d/%d served predictions from closed positions",
                len(matched), len(preds))
    if matched.empty:
        return pd.DataFrame()

    keep_meta = [c for c in META_COLS if c in matched.columns]
    out = matched[keep_meta + feat_names].copy()
    out["position_id"] = out["position_id"].astype(int)
    return out


def drift_audit(labeled_prod: pd.DataFrame, cache: KlineCache | None = None) -> None:
    """Compare each labeled prediction's SERVED features to features recomputed from the
    cache at the same bar. Reports max abs drift (serve/train consistency)."""
    if labeled_prod.empty:
        logger.info("drift audit: no labeled production rows")
        return
    cache = cache or KlineCache()
    try:
        fg = FeatureGenerator(btc=cache.load_btc())
    except FileNotFoundError:
        logger.warning("drift audit skipped: BTC not in cache")
        return
    feats = feature_cols(labeled_prod)
    diffs, audited = [], 0
    for token, grp in labeled_prod.groupby("token"):
        try:
            frame = fg.generate(cache.load(token))
        except FileNotFoundError:
            continue
        lo, hi = frame.index.min(), frame.index.max()
        for r in grp.itertuples(index=False):
            ts = pd.Timestamp(r.opened_at, tz="UTC").floor("min") - pd.Timedelta(minutes=FEATURE_LAG_MIN)
            if ts < lo or ts > hi:
                continue
            recomputed = fg.sample_at(frame, [ts]).drop(columns=["event_time"]).iloc[0]
            served = pd.Series({f: getattr(r, f) for f in feats})
            d = (served.astype(float) - recomputed[feats].astype(float)).abs()
            diffs.append(d.max()); audited += 1
    if diffs:
        logger.info("drift audit: %d bars audited, max feature drift %.3e (median %.3e)",
                    audited, float(pd.Series(diffs).max()), float(pd.Series(diffs).median()))
    else:
        logger.info("drift audit: no cache-covered bars to audit (all predictions too recent)")


def _reload_service(uri=SERVICE_URI) -> None:
    try:
        r = requests.post(f"{uri}/reload", timeout=30)
        r.raise_for_status()
        logger.info("service reloaded: %s", r.json())
    except requests.RequestException as exc:
        logger.warning("service /reload failed (non-fatal): %s", exc)


def run(skip_reload: bool = False) -> dict:
    # dataset_csv owns ALL feature sourcing (cached csv -> feature store -> Binance REST).
    # Do not union label_feature_log() back in here: the csv already incorporates the
    # served rows, and re-unioning would double-source positions AND invert the precedence
    # (a REST-recomputed row would win over the vector production actually served).
    combined = dataset_csv.sync()
    logger.info("training set: %d rows | by source: %s",
                len(combined), combined[dataset_csv.SOURCE_COL].value_counts().to_dict())

    summary = train_mod.train(combined.drop(columns=[dataset_csv.SOURCE_COL]))
    logger.info("retrain: v%s walk_forward_auc=%.4f promoted=%s",
                summary["version"], summary["walk_forward_auc"], summary["promoted"])

    if summary["promoted"] and not skip_reload:
        _reload_service()

    # audit only — label_feature_log is never a training source now
    drift_audit(label_feature_log())
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-reload", action="store_true")
    args = ap.parse_args()
    summary = run(skip_reload=args.skip_reload)
    print("\n" + "=" * 64)
    print(f"retrain -> {train_mod.REGISTERED_MODEL} v{summary['version']}  "
          f"walk_forward_auc={summary['walk_forward_auc']:.4f}  promoted={summary['promoted']}")
    print(f"gate: {summary['gate_reason']}")
    print("=" * 64)


if __name__ == "__main__":
    main()

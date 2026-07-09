"""dataset.csv — the durable, incrementally-grown training set. The ONLY feature-sourcing
path for retraining.

The igni parquet cache is a static snapshot that stops at its last cached bar; positions
opened after that were silently dropped by `dataset.build()` (a bare `continue` on missing
kline coverage), so retrain slowly went blind to recent history. This module fixes that by
caching every position's features once, then backfilling only what is missing.

Sourcing precedence for a position not already in the csv:
  1. CSV      already cached -> never recomputed (features are frozen at write time)
  2. STORE    the vector actually SERVED in production (feature_store.db) -> best
              train/serve consistency; used whenever the bot logged a save=true prediction
  3. REST     Binance historical klines -> recompute features from scratch

`source` records which path produced each row, so the provenance stays auditable.

Two consequences, both deliberate:
  * dataset.csv is irreplaceable — if lost while the kline cache is stale, recent rows can
    only be rebuilt by refetching. BACK UP the data/ directory.
  * cached rows freeze the featuregen.py that produced them. `check_meta()` fails fast on a
    featuregen change rather than silently mixing two feature definitions in one training
    set. Editing featuregen.py means `python dataset_csv.py --rebuild`.

    python dataset_csv.py --rebuild    # bootstrap, or after editing featuregen.py
    python dataset_csv.py --sync       # backfill missing positions (what retrain calls)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

import dataset as dataset_mod
from data_sources import KlineCache, load_closed_positions
from dataset import FEATURE_LAG_MIN, attach_target
from featuregen import FeatureGenerator
from model import META_COLS
from serving.binance_klines import fetch_range_1m_sync

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("SLP_DATASET_DIR",
                               str(Path(__file__).resolve().parent / "data")))
CSV_PATH = DATA_DIR / "dataset.csv"
META_PATH = DATA_DIR / "dataset.csv.meta.json"
SKIP_PATH = DATA_DIR / "dataset.csv.skip.json"

FEATUREGEN_PATH = Path(__file__).resolve().parent / "featuregen.py"
SOURCE_COL = "source"
DATE_COLS = ["opened_at", "closed_at", "intended_close"]

# Warmup bars fetched before a sample bar. The longest rolling window is 240; beta/corr_btc
# need 120; pct_change needs 1 prior. 300 gives headroom over the >=241 minimum.
WARMUP_MIN = 300
BTC_SYMBOL = "BTC/USDT"

# A transient REST failure is retried on later syncs; give up after this many attempts so a
# genuinely broken position can't stall every run forever.
MAX_SKIP_ATTEMPTS = 5


# --------------------------------------------------------------------------- #
# meta sidecar — fail fast when featuregen.py changed under a cached csv
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _names_sha256(names) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def _feature_names(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS and c != SOURCE_COL]


def write_meta(df: pd.DataFrame) -> None:
    names = _feature_names(df)
    META_PATH.write_text(json.dumps({
        "featuregen_sha256": _sha256(FEATUREGEN_PATH),
        "feature_names_sha256": _names_sha256(names),
        "n_features": len(names),
        "n_rows": len(df),
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, indent=2) + "\n")


def check_meta() -> None:
    """Raise if featuregen.py changed since dataset.csv was built. Cached rows carry the
    OLD feature definitions; newly fetched rows would carry the new ones."""
    if not CSV_PATH.exists():
        return                                   # nothing cached yet; --rebuild will write
    if not META_PATH.exists():
        raise RuntimeError(
            f"{CSV_PATH} exists but {META_PATH.name} is missing — cannot verify the feature "
            f"definitions it was built with. Rebuild: python dataset_csv.py --rebuild")
    meta = json.loads(META_PATH.read_text())
    current = _sha256(FEATUREGEN_PATH)
    if meta.get("featuregen_sha256") != current:
        raise RuntimeError(
            f"featuregen.py changed since dataset.csv was built "
            f"({meta.get('featuregen_sha256', '?')[:12]} != {current[:12]}). Cached rows "
            f"hold the old feature definitions; mixing them with newly computed rows would "
            f"silently corrupt the training set.\n"
            f"    Rebuild: python dataset_csv.py --rebuild")


# --------------------------------------------------------------------------- #
# load / save
# --------------------------------------------------------------------------- #
def load() -> pd.DataFrame:
    """float_precision='round_trip' is required, not cosmetic: the default 'high' parser is
    ~1 ulp lossy on values needing 17 significant digits, so read->write would shave a digit
    off cached features on EVERY sync. Cached rows must be bit-stable once written."""
    if not CSV_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(CSV_PATH, float_precision="round_trip")
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


def save_atomic(df: pd.DataFrame) -> None:
    """Write via tmp + os.replace so a crash mid-write can never truncate the dataset.
    NOTE: this swaps the inode — the csv must live in a bind-mounted DIRECTORY, never be
    bind-mounted as a single file, or the container keeps reading the stale inode."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CSV_PATH.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, CSV_PATH)
    write_meta(df)
    logger.info("wrote %d rows -> %s", len(df), CSV_PATH)


def _load_skip() -> dict:
    return json.loads(SKIP_PATH.read_text()) if SKIP_PATH.exists() else {}


def _record_skip(skip: dict, position_id: int, reason: str, permanent: bool) -> None:
    key = str(position_id)
    prev = skip.get(key, {})
    skip[key] = {"reason": reason, "permanent": permanent,
                 "attempts": prev.get("attempts", 0) + 1,
                 "last_tried": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def _permanently_skipped(skip: dict) -> set[int]:
    """position_ids to stop retrying. A delisted token or a genuinely absent bar will never
    resolve. A network blip WILL — retrying it (up to MAX_SKIP_ATTEMPTS) is what keeps a
    transient Binance failure from silently benching those positions forever, which would
    reintroduce the very blindness dataset.csv exists to prevent."""
    out = set()
    for pid, rec in skip.items():
        if rec.get("permanent") or rec.get("attempts", 0) >= MAX_SKIP_ATTEMPTS:
            out.add(int(pid))
    return out


def _save_skip(skip: dict) -> None:
    if skip:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SKIP_PATH.write_text(json.dumps(skip, indent=2) + "\n")


# --------------------------------------------------------------------------- #
# diff
# --------------------------------------------------------------------------- #
def missing_positions(positions: pd.DataFrame, csv: pd.DataFrame) -> pd.DataFrame:
    """Closed positions with no row in the csv. `positions` MUST already be post-
    attach_target: it drops early-but-profitable rows, and diffing against the raw set
    would mark those missing forever and refetch them on every run."""
    have = set(csv["position_id"].astype(int)) if not csv.empty else set()
    skip = _permanently_skipped(_load_skip())
    miss = positions[~positions["position_id"].astype(int).isin(have | skip)]
    if skip:
        logger.info("%d position(s) permanently skipped (see %s)", len(skip), SKIP_PATH.name)
    return miss


def _sample_ts(opened_at) -> pd.Timestamp:
    return pd.Timestamp(opened_at, tz="UTC").floor("min") - pd.Timedelta(minutes=FEATURE_LAG_MIN)


# --------------------------------------------------------------------------- #
# source 2: the served feature vectors
# --------------------------------------------------------------------------- #
def fill_from_store(missing: pd.DataFrame) -> pd.DataFrame:
    """Rows recoverable from feature_store.db — the exact vectors production served.
    Reuses retrain_monthly.label_feature_log() (merge_asof on token, 5min tolerance)."""
    if missing.empty:
        return pd.DataFrame()
    from retrain_monthly import label_feature_log      # local: avoids a circular import

    labeled = label_feature_log()
    if labeled.empty:
        return pd.DataFrame()
    want = set(missing["position_id"].astype(int))
    hit = labeled[labeled["position_id"].astype(int).isin(want)].copy()
    if not hit.empty:
        hit[SOURCE_COL] = "store"
        logger.info("filled %d position(s) from the feature store", len(hit))
    return hit


# --------------------------------------------------------------------------- #
# source 3: Binance historical REST
# --------------------------------------------------------------------------- #
def fill_from_rest(missing: pd.DataFrame, skip: dict) -> pd.DataFrame:
    """Recompute features from Binance historical klines. One paged range per token; BTC
    fetched ONCE for the whole span and reused across every token."""
    if missing.empty:
        return pd.DataFrame()

    missing = missing.copy()
    missing["sample_ts"] = missing["opened_at"].map(_sample_ts)
    lo = missing["sample_ts"].min() - pd.Timedelta(minutes=WARMUP_MIN)
    hi = missing["sample_ts"].max()

    logger.info("REST backfill: %d position(s), %d token(s), span %s -> %s",
                len(missing), missing["token"].nunique(), lo, hi)
    btc = fetch_range_1m_sync(BTC_SYMBOL, lo, hi)
    if btc.empty:
        raise RuntimeError("BTC range fetch returned no bars — cannot compute BTC features")
    fg = FeatureGenerator(btc=btc)

    rows = []
    for token, grp in missing.groupby("token"):
        t_lo = grp["sample_ts"].min() - pd.Timedelta(minutes=WARMUP_MIN)
        try:
            df1m = fetch_range_1m_sync(f"{token}/USDT", t_lo, grp["sample_ts"].max())
        except requests.RequestException as exc:
            # transient by default (network/5xx/timeout) -> retried next sync. An unknown or
            # delisted symbol answers 400, which is permanent.
            permanent = getattr(exc.response, "status_code", None) == 400
            for r in grp.itertuples(index=False):
                _record_skip(skip, r.position_id, f"kline fetch failed: {exc}", permanent)
            logger.warning("%s: fetch failed (%s) — %d position(s) skipped (permanent=%s)",
                           token, exc, len(grp), permanent)
            continue
        if df1m.empty:
            for r in grp.itertuples(index=False):
                _record_skip(skip, r.position_id, "no klines (unknown/delisted symbol)", True)
            logger.warning("%s: no klines — %d position(s) skipped", token, len(grp))
            continue

        frame = fg.generate(df1m)
        for r in grp.itertuples(index=False):
            ts = r.sample_ts
            # sample_at ffills: a missing bar silently yields an EARLIER bar's features.
            # Require the exact bar rather than write a wrong row.
            if ts not in frame.index:
                _record_skip(skip, r.position_id, f"no bar at sample_ts {ts}", True)
                continue
            feat = fg.sample_at(frame, [ts]).drop(columns=["event_time"]).iloc[0].to_dict()
            rows.append({
                "position_id": r.position_id, "token": token, "symbol": r.symbol,
                "opened_at": r.opened_at, "closed_at": r.closed_at,
                "funding_period": r.funding_period, "funding_rate": r.funding_rate,
                "entry_price": r.entry_price, "intended_close": r.intended_close,
                "margin_min": r.margin_min, "realized_pnl": r.realized_pnl,
                "target": r.target, **feat, SOURCE_COL: "rest",
            })

    out = pd.DataFrame(rows)
    logger.info("REST backfill produced %d row(s)", len(out))
    return out


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def _order(df: pd.DataFrame) -> pd.DataFrame:
    """META first, then features, then source — the deterministic column order feats.json
    downstream depends on."""
    meta = [c for c in META_COLS if c in df.columns]
    feats = [c for c in df.columns if c not in META_COLS and c != SOURCE_COL]
    return df[meta + feats + [SOURCE_COL]]


def sync(rebuild: bool = False) -> pd.DataFrame:
    """existing csv -> feature store -> REST. Appends, dedups, saves, returns the dataset."""
    positions = attach_target(load_closed_positions())
    logger.info("%d closed binance positions (post-filter)", len(positions))

    if rebuild:
        logger.info("rebuild: regenerating cache-covered rows from the kline cache")
        csv = dataset_mod.build_dataset(positions, KlineCache())
        if not csv.empty:
            csv[SOURCE_COL] = "cache"
        if SKIP_PATH.exists():
            SKIP_PATH.unlink()                   # a rebuild retries everything
    else:
        check_meta()
        csv = load()
    logger.info("starting from %d cached row(s)", len(csv))

    skip = {} if rebuild else _load_skip()
    missing = missing_positions(positions, csv)
    logger.info("%d position(s) missing features", len(missing))

    parts = [csv] if not csv.empty else []
    if not missing.empty:
        from_store = fill_from_store(missing)
        if not from_store.empty:
            parts.append(from_store)
            done = set(from_store["position_id"].astype(int))
            missing = missing[~missing["position_id"].astype(int).isin(done)]

        from_rest = fill_from_rest(missing, skip)
        if not from_rest.empty:
            parts.append(from_rest)

    _save_skip(skip)
    if not parts:
        raise RuntimeError("no training rows — check the kline cache and funding db")

    out = pd.concat(parts, ignore_index=True).drop_duplicates(
        subset="position_id", keep="first")
    out["position_id"] = out["position_id"].astype(int)
    out = _order(out).sort_values("opened_at").reset_index(drop=True)
    save_atomic(out)

    covered = len(out) / len(positions) * 100 if len(positions) else 0.0
    logger.info("dataset.csv: %d/%d positions (%.1f%%) | by source: %s",
                len(out), len(positions), covered,
                out[SOURCE_COL].value_counts().to_dict())
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rebuild", action="store_true",
                   help="rebuild every row from scratch (bootstrap / after featuregen change)")
    g.add_argument("--sync", action="store_true", help="backfill only the missing positions")
    args = ap.parse_args()

    ds = sync(rebuild=args.rebuild)
    print("\n" + "=" * 64)
    print(f"dataset.csv -> {len(ds)} rows x {ds.shape[1]} cols")
    print(f"sources: {ds[SOURCE_COL].value_counts().to_dict()}")
    print(f"target=1 base rate: {ds['target'].mean() * 100:.1f}%")
    print("=" * 64)


if __name__ == "__main__":
    main()

"""Build the labeled training set: CLOSED Binance positions -> features@(open-1min)
-> SL target. Owns the target definition and the leak-free sampling contract.

  1. POSITIONS  every CLOSED Binance position (data_sources.load_closed_positions).
  2. TARGET     1 = "SL hit" = closed BEFORE the intended close, where the intended
                close is CLOSE_LEAD_MIN(=30) min before the NEXT funding payment. The
                intended close is anchored to the funding SCHEDULE grid (opened_at
                floored to the fp-hour grid + fp*h - 30min), NOT to opened_at itself:
                the bot opens ~5s after the payment, so an opened_at anchor would
                mislabel normal closes that fire a few seconds early as SL hits.
                Early-but-PROFITABLE closes are dropped (favorable early exits, not the
                loss-making stop-out the label captures -> ambiguous).
  3. FEATURES   FeatureGenerator (65 causal feats) sampled at the last CLOSED 1m bar
                before the position opened: open_ts.floor('min') - FEATURE_LAG_MIN(=1)
                min -> no leak. Positions with no kline coverage at that bar are dropped.

    python dataset.py                       # build -> position_dataset.parquet
    python dataset.py --out foo.parquet
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from data_sources import KlineCache, load_closed_positions
from featuregen import FeatureGenerator
from model import META_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLOSE_LEAD_MIN = 30           # bot closes 30 min before the next funding payment
FEATURE_LAG_MIN = 1           # sample features at open - 1min (last fully closed bar)


def attach_target(positions: pd.DataFrame) -> pd.DataFrame:
    """Add intended_close / margin_min / target, then drop ambiguous early-profitable
    rows. Funding-grid anchored so open-lag does not mislabel normal closes."""
    df = positions.copy()
    # intended close anchored to the funding schedule grid (robust to open lag)
    payment = df.apply(lambda r: r["opened_at"].floor(f"{int(r['funding_period'])}h"), axis=1)
    df["intended_close"] = (payment + pd.to_timedelta(df["funding_period"], "h")
                            - pd.Timedelta(minutes=CLOSE_LEAD_MIN))
    df["margin_min"] = (df["closed_at"] - df["intended_close"]).dt.total_seconds() / 60.0
    df["target"] = (df["closed_at"] < df["intended_close"]).astype(int)

    ambiguous = (df["target"] == 1) & (df["realized_pnl"] > 0)
    logger.info("dropping %d early-but-profitable positions (ambiguous label)",
                int(ambiguous.sum()))
    return df[~ambiguous].reset_index(drop=True)


def build_dataset(positions: pd.DataFrame, cache: KlineCache) -> pd.DataFrame:
    """One labeled (META_COLS + 65 features) row per position that has klines at
    open-1min. `positions` must already carry the target (see attach_target)."""
    btc = cache.load_btc()
    fg = FeatureGenerator(btc=btc)

    rows, tokens = [], positions["token"].unique()
    for i, token in enumerate(sorted(tokens), 1):
        try:
            df1m = cache.load(token)
        except FileNotFoundError:
            continue
        frame = fg.generate(df1m)
        lo, hi = frame.index.min(), frame.index.max()

        pos = positions[positions["token"] == token]
        for r in pos.itertuples(index=False):
            open_ts = pd.Timestamp(r.opened_at, tz="UTC")
            sample_ts = open_ts.floor("min") - pd.Timedelta(minutes=FEATURE_LAG_MIN)
            if sample_ts < lo or sample_ts > hi:      # no feature data at open
                continue
            feat = fg.sample_at(frame, [sample_ts]).drop(columns=["event_time"]).iloc[0].to_dict()
            rows.append({
                "position_id": r.position_id, "token": token, "symbol": r.symbol,
                "opened_at": r.opened_at, "closed_at": r.closed_at,
                "funding_period": r.funding_period, "funding_rate": r.funding_rate,
                "entry_price": r.entry_price, "intended_close": r.intended_close,
                "margin_min": r.margin_min, "realized_pnl": r.realized_pnl,
                "target": r.target, **feat,
            })
        if i % 50 == 0:
            logger.info("processed %d/%d tokens, %d labeled positions", i, len(tokens), len(rows))

    out = pd.DataFrame(rows)
    if not out.empty:
        # META first, then features (deterministic column order for feats.json downstream)
        feat_cols = [c for c in out.columns if c not in META_COLS]
        out = out[[c for c in META_COLS if c in out.columns] + feat_cols]
        logger.info("DONE: %d labeled positions | target=1 (SL) base rate %.1f%%",
                    len(out), out["target"].mean() * 100)
    return out


def build(db_path=None, cache_dir=None) -> pd.DataFrame:
    """Full build: load -> label -> features. Returns the labeled DataFrame."""
    positions = load_closed_positions() if db_path is None else load_closed_positions(db_path)
    positions = attach_target(positions)
    logger.info("%d closed binance positions (post-filter) | target=1 base rate %.1f%%",
                len(positions), positions["target"].mean() * 100)
    cache = KlineCache() if cache_dir is None else KlineCache(cache_dir)
    return build_dataset(positions, cache)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="position_dataset.parquet")
    args = ap.parse_args()

    ds = build()
    if ds.empty:
        logger.warning("no labeled positions built (check kline cache coverage)")
        return
    ds.to_parquet(args.out)
    logger.info("wrote %d rows x %d cols -> %s", len(ds), ds.shape[1], args.out)


if __name__ == "__main__":
    main()

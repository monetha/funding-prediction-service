"""Raw training inputs: kline history + position labels. No feature or target logic
here (that lives in dataset.py) — this module only *reads*.

Two sources, both external to this repo on the training host (override via env):
  * KlineCache  — the igni parquet cache of raw 1m Binance-futures klines with taker
                  flow (open, high, low, close, volume, quote_vol, trades, tbb[, tbq]),
                  one file per token + a BTC reference. Env: SLP_KLINE_CACHE.
  * load_closed_positions — CLOSED Binance positions from the funding bot's sqlite db,
                  the label source. Env: SLP_FUNDING_DB.

The service OWNS this file going forward; there is no import link back to the source
funding-trader-bot repo.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

# Defaults point at this host's data; override with env vars for other environments.
# The validated training set (walk-forward AUC 0.654) was built from this cache.
DEFAULT_KLINE_CACHE = Path(os.environ.get(
    "SLP_KLINE_CACHE", "/root/funding-trader-bot/feature_mining/data"))
DEFAULT_FUNDING_DB = Path(os.environ.get("SLP_FUNDING_DB",
                                         "/root/funding-trader-bot/db/funding_bot.db"))
BTC_TOKEN = "BTC"


class KlineCache:
    """Reader for the igni parquet cache — the data source for FeatureGenerator.

    Files: `fut6m__<TOKEN>_USDT-USDT.parquet` (raw 1m klines with taker flow), UTC
    DatetimeIndex, ~6 months. That schema is exactly FeatureGenerator.REQUIRED_COLS
    (+ tbq). BTC is loaded once and reused as the regime reference across tokens.
    """

    def __init__(self, cache_dir: Path | str = DEFAULT_KLINE_CACHE):
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(f"kline cache dir not found: {self.cache_dir}")

    def _stem_to_token(self, stem: str) -> str:
        # "fut6m__BARD_USDT-USDT" -> "BARD"
        return stem.removeprefix("fut6m__").split("_USDT-USDT")[0]

    def list_tokens(self, include_btc: bool = False) -> list[str]:
        """All tokens with a fut6m parquet. BTC excluded by default — it's the regime
        *reference*, and token-vs-itself makes rel_strength/corr degenerate."""
        toks = sorted(self._stem_to_token(p.stem)
                      for p in self.cache_dir.glob("fut6m__*.parquet"))
        if not include_btc:
            toks = [t for t in toks if t != BTC_TOKEN]
        return toks

    def _path(self, token: str) -> Path:
        p = self.cache_dir / f"fut6m__{token}_USDT-USDT.parquet"
        if not p.exists():
            raise FileNotFoundError(f"no fut6m cache for token {token!r}: {p}")
        return p

    def load(self, token: str) -> pd.DataFrame:
        """Raw 1m futures klines for a token (ready for FeatureGenerator.generate)."""
        return _utc(pd.read_parquet(self._path(token)))

    def load_btc(self) -> pd.DataFrame:
        """BTC reference frame (load once, pass to FeatureGenerator(btc=...))."""
        return self.load(BTC_TOKEN)


def load_closed_positions(db_path: Path | str = DEFAULT_FUNDING_DB) -> pd.DataFrame:
    """All CLOSED Binance positions as RAW rows (no target computed here). Columns:
    position_id, symbol, token, side, entry_price, opened_at, closed_at,
    funding_period, funding_rate, realized_pnl. Timestamps parsed to tz-naive datetimes
    (dataset.py localizes to UTC when anchoring the funding grid)."""
    con = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        "SELECT id AS position_id, symbol, side, entry_price, opened_at, closed_at, "
        "funding_period, funding_rate, realized_pnl FROM positions "
        "WHERE exchange='binance' AND status='CLOSED'",
        con,
    )
    con.close()
    df["opened_at"] = pd.to_datetime(df["opened_at"])
    df["closed_at"] = pd.to_datetime(df["closed_at"])
    df["token"] = df["symbol"].str.split("/").str[0]
    return df


def _utc(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a tz-aware (UTC) DatetimeIndex so FeatureGenerator.sample_at aligns."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("kline frame must have a DatetimeIndex (1m UTC bars)")
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    return df


if __name__ == "__main__":
    cache = KlineCache()
    toks = cache.list_tokens()
    print(f"{len(toks)} tokens (BTC excluded) in {cache.cache_dir}")
    pos = load_closed_positions()
    print(f"{len(pos)} closed binance positions in {DEFAULT_FUNDING_DB}")
    print(pos[["position_id", "token", "opened_at", "closed_at",
               "funding_period", "realized_pnl"]].head())

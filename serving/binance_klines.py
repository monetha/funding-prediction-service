"""Raw 1m klines from Binance USDS-M futures public REST (no API key).

    GET https://fapi.binance.com/fapi/v1/klines?symbol=<SYM>&interval=1m&limit=300

returns arrays of 12 fields; we map the 8 that FeatureGenerator needs (incl. taker
buy base volume `tbb`, quote volume, trade count — which exist ONLY in raw klines).

Key serving contract:
  * symbol mapping: CCXT unified 'SOL/USDT:USDT' -> Binance 'SOLUSDT'.
  * DROP the in-progress last bar, but ONLY when it really is in progress (its open time
    is the current minute) — Binance creates a bar only once a trade occurs, so an
    illiquid symbol's last element early in a new minute is already closed. Either way
    the resulting last bar's open time is now.floor('min') - 1min == the sample bar
    (matches training FEATURE_LAG_MIN=1).
  * limit=300: longest rolling window is 240 bars, beta/corr_btc need 120, pct_change
    needs 1 prior -> need >=241; 300 gives headroom after dropping the partial bar.
"""
from __future__ import annotations

import os

import aiohttp
import pandas as pd
import requests

BASE_URL = os.environ.get("SLP_BINANCE_FAPI", "https://fapi.binance.com")
KLINES_PATH = "/fapi/v1/klines"
DEFAULT_LIMIT = 300
MAX_LIMIT = 1500                # per-request cap for the paged historical range fetch

# Binance kline array indices -> our column names (the 8 FeatureGenerator needs).
# 0 openTime | 1 open | 2 high | 3 low | 4 close | 5 volume | 6 closeTime |
# 7 quoteAssetVolume | 8 numberOfTrades | 9 takerBuyBaseVolume | 10 takerBuyQuote | 11 ignore
_COLS = ["open", "high", "low", "close", "volume", "quote_vol", "trades", "tbb"]
_IDX = {"open": 1, "high": 2, "low": 3, "close": 4, "volume": 5,
        "quote_vol": 7, "trades": 8, "tbb": 9}


def _is_partial(bar: list) -> bool:
    """Is this kline the still-forming current minute?

    Binance only creates a bar once a trade occurs in it, so the last element is NOT
    always partial: for an illiquid symbol in the first seconds of a new minute the bar
    does not exist yet and the last element is the newest CLOSED bar. Dropping it
    unconditionally discarded that bar, and FeatureGenerator.sample_at's ffill then
    silently served the minute BEFORE the requested one (see experiments/FINDINGS.md).
    Compare the open time to the current minute instead of trusting position.
    """
    return pd.to_datetime(bar[0], unit="ms", utc=True) == pd.Timestamp.now(tz="UTC").floor("min")


def to_binance_symbol(symbol: str) -> str:
    """CCXT unified / loose forms -> Binance futures symbol.
    'SOL/USDT:USDT' -> 'SOLUSDT'; 'SOL/USDT' -> 'SOLUSDT'; 'SOLUSDT' -> 'SOLUSDT'."""
    s = symbol.strip().upper()
    s = s.split(":", 1)[0]          # drop settlement suffix ':USDT'
    s = s.replace("/", "")          # 'SOL/USDT' -> 'SOLUSDT'
    return s


def parse_klines(raw: list, drop_partial: bool = True) -> pd.DataFrame:
    """Binance klines array -> DataFrame(open..tbb) on a UTC 1m DatetimeIndex.
    Drops the in-progress last bar by default (see module docstring)."""
    if not raw:
        return pd.DataFrame(columns=_COLS)
    if drop_partial and _is_partial(raw[-1]):
        raw = raw[:-1]              # last element is the current, still-forming minute
    idx = pd.to_datetime([r[0] for r in raw], unit="ms", utc=True)
    data = {c: [float(r[_IDX[c]]) for r in raw] for c in _COLS}
    return pd.DataFrame(data, index=idx)


def _params(symbol: str, limit: int, start_ms: int | None = None,
            end_ms: int | None = None) -> dict:
    p = {"symbol": to_binance_symbol(symbol), "interval": "1m", "limit": limit}
    if start_ms is not None:
        p["startTime"] = start_ms
    if end_ms is not None:
        p["endTime"] = end_ms
    return p


async def fetch_raw_1m(session: aiohttp.ClientSession, symbol: str,
                       limit: int = DEFAULT_LIMIT, drop_partial: bool = True,
                       timeout: float = 8.0) -> pd.DataFrame:
    """Async fetch + parse. Raises aiohttp.ClientError / asyncio.TimeoutError on failure
    (callers fail-closed). Returns closed 1m bars, newest last."""
    async with session.get(BASE_URL + KLINES_PATH, params=_params(symbol, limit),
                           timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
        resp.raise_for_status()
        raw = await resp.json()
    return parse_klines(raw, drop_partial=drop_partial)


def fetch_raw_1m_sync(symbol: str, limit: int = DEFAULT_LIMIT,
                      drop_partial: bool = True, timeout: float = 8.0) -> pd.DataFrame:
    """Blocking fetch (for tests / offline scripts / retrain drift audit)."""
    resp = requests.get(BASE_URL + KLINES_PATH, params=_params(symbol, limit), timeout=timeout)
    resp.raise_for_status()
    return parse_klines(resp.json(), drop_partial=drop_partial)


def fetch_range_1m_sync(symbol: str, start: pd.Timestamp, end: pd.Timestamp,
                        timeout: float = 15.0) -> pd.DataFrame:
    """Blocking historical fetch of closed 1m bars over [start, end], paging at
    MAX_LIMIT. Used to backfill features for positions the kline cache never covered.

    drop_partial=False: every bar in a past window is already closed, and dropping the
    last element would silently lose the sample bar when `end` is the sample timestamp.
    An unknown/delisted symbol answers HTTP 400 -> raises (callers classify that as a
    permanent skip). An empty body for a valid symbol returns an empty frame.
    """
    frames, cur = [], start
    while cur <= end:
        resp = requests.get(
            BASE_URL + KLINES_PATH,
            params=_params(symbol, MAX_LIMIT,
                           start_ms=int(cur.timestamp() * 1000),
                           end_ms=int(end.timestamp() * 1000)),
            timeout=timeout)
        resp.raise_for_status()
        raw = resp.json()
        if not raw:
            break
        frames.append(parse_klines(raw, drop_partial=False))
        last = pd.to_datetime(raw[-1][0], unit="ms", utc=True)
        if last >= end or len(raw) < MAX_LIMIT:
            break
        cur = last + pd.Timedelta(minutes=1)      # next page starts after the last bar
    if not frames:
        return pd.DataFrame(columns=_COLS)
    out = pd.concat(frames)
    return out[~out.index.duplicated()].sort_index()

"""Causal feature generator — the OWNED, single source of train/serve parity.

This is a vendored copy of the proven futures-era feature engine (65 causal
features: rich price/vol + taker buy/sell FLOW + trade-count + BTC market regime).
Every column is computed as rolling/shift only, so it is **leak-free by
construction** — column at bar t uses only bars <= t. You then *sample* the row at
the event bar, and it can never see the future.

THIS EXACT FILE is imported by BOTH training (dataset.py/train.py) and serving
(serving/app.py). Do not fork it — parity depends on one implementation. The
authoritative feature set + column order is whatever `feature_cols()` (model.py)
derives from a frame generated here, pinned into `feats.json` at train time.

Input schema (raw Binance USDS-M klines — `fapiPublicGetKlines`, 12 fields)
--------------------------------------------------------------------------
Requires a frame with columns:

    open, high, low, close, volume, quote_vol, trades, tbb   (tbq optional)

on a UTC DatetimeIndex of 1-minute bars. The flagship flow features
(`flow_imb_*`, `buy_ratio_*`, `cvd_slope_*`, `trades_z_60`, `avg_trade_size_*`)
need taker buy base volume (`tbb`, field [9]), quote volume and trade count — these
exist ONLY in raw klines, not standard ccxt `fetch_ohlcv` (6 fields).

BTC frame (columns: close, tbb, volume) drives the 8 `btc_*` regime features. When
omitted those features are skipped (57 total instead of 65) — serving MUST always
supply BTC so the model always sees the exact 65-feature input it was trained on.

Usage: build the frame once per token, then `sample_at(frame, event_times)` to pull
the event rows — the "raw klines -> features" contract for both train and serve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12

# Input schema (igni-bot data_fut.fetch_fut_klines). tbq is accepted but unused here.
REQUIRED_COLS = ("open", "high", "low", "close", "volume", "quote_vol", "trades", "tbb")
BTC_REQUIRED_COLS = ("close", "tbb", "volume")


class FeatureGenerator:
    """Generate the causal feature frame for a single token's 1m futures klines.

    Parameters
    ----------
    btc : optional BTC reference frame (columns: close, tbb, volume; 1m UTC index).
        When omitted, the ~8 `btc_*` regime features are skipped (logged via the
        returned column set), not errored.

    The rolling windows match igni-bot's proven set and are exposed as attributes
    only for inspection — change them deliberately, the defaults are the validated
    configuration.
    """

    # window sets (1m bars) — igni-bot's proven configuration
    RET_WINDOWS = (1, 3, 5, 10, 15, 30, 60, 120, 240)
    RV_WINDOWS = (15, 30, 60, 240)
    POS_WINDOWS = (15, 30, 60)
    DRAWDOWN_WINDOWS = (60, 240)
    VWAP_WINDOWS = (30, 120)
    FLOW_WINDOWS = (5, 15, 60)

    def __init__(self, btc: pd.DataFrame | None = None):
        self.btc = btc
        if btc is not None:
            self._validate(btc, BTC_REQUIRED_COLS, "btc frame")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate(df: pd.DataFrame, cols, name: str) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"{name} missing required column(s) {missing}; need {tuple(cols)}. "
                "Taker-flow columns (tbb/quote_vol/trades) come only from raw "
                "Binance klines (fapiPublicGetKlines), not standard ccxt fetch_ohlcv."
            )
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{name} must have a DatetimeIndex (1m UTC bars)")

    @staticmethod
    def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
        d = close.diff()
        up = d.clip(lower=0).rolling(n).mean()
        dn = (-d.clip(upper=0)).rolling(n).mean()
        return 100 - 100 / (1 + up / (dn + EPS))

    # ------------------------------------------------------------------ #
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """1m kline frame -> causal feature frame (same index). Leak-free by
        construction (rolling/shift only look backward)."""
        self._validate(df, REQUIRED_COLS, "input frame")
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        v, qv, trd = df["volume"], df["quote_vol"], df["trades"]
        buy = df["tbb"]                         # taker buy base volume
        sell = (v - buy).clip(lower=0)
        r1 = c.pct_change()
        F: dict[str, pd.Series] = {}

        self._returns_block(F, c, r1)
        self._candle_block(F, o, h, l, c, v, r1)
        self._volume_block(F, v, qv, r1)
        self._flow_block(F, buy, sell, v)
        self._trades_block(F, qv, trd)
        if self.btc is not None:
            self._btc_block(F, c, r1)
        self._time_block(F, c.index)

        return pd.DataFrame(F, index=c.index).replace([np.inf, -np.inf], np.nan)

    # ----- feature blocks (formulas verbatim from igni-bot) ------------ #
    def _returns_block(self, F, c, r1):
        rets = {}
        for w in self.RET_WINDOWS:
            rets[w] = c / c.shift(w) - 1.0
            F[f"ret_{w}"] = rets[w]
        F["accel_5_15"] = rets[5] - rets[15]
        F["accel_15_60"] = rets[15] - rets[60]
        rv = {}
        for w in self.RV_WINDOWS:
            rv[w] = r1.rolling(w).std()
            F[f"rv_{w}"] = rv[w]
        F["rv_expand"] = rv[15] / (rv[240] + EPS)
        F["mom_norm_15"] = rets[15] / (rv[60] + EPS)
        F["mom_norm_60"] = rets[60] / (rv[240] + EPS)
        F["ret_skew_60"] = r1.rolling(60).skew()
        F["ret_kurt_60"] = r1.rolling(60).kurt()
        self._rets = rets  # reused by btc rel_strength

    def _candle_block(self, F, o, h, l, c, v, r1):
        rng = (h - l).abs() + EPS
        F["body_15"] = ((c - o) / rng).rolling(15).mean()
        F["uwick_15"] = ((h - np.maximum(o, c)) / rng).rolling(15).mean()
        F["lwick_15"] = ((np.minimum(o, c) - l) / rng).rolling(15).mean()
        F["green_frac_15"] = (c > o).astype(float).rolling(15).mean()
        F["hl_range_30"] = (rng / c).rolling(30).mean()
        for w in self.POS_WINDOWS:
            hi = h.rolling(w).max(); lo = l.rolling(w).min()
            F[f"pos_in_range_{w}"] = (c - lo) / (hi - lo + EPS)
        for w in self.DRAWDOWN_WINDOWS:
            F[f"drawdown_{w}"] = c / h.rolling(w).max() - 1.0
        for w in self.VWAP_WINDOWS:
            vwap = (c * v).rolling(w).sum() / (v.rolling(w).sum() + EPS)
            F[f"vwap_dev_{w}"] = c / vwap - 1.0
        F["rsi_14"] = self._rsi(c, 14)
        up = (r1 > 0).astype(int)
        F["up_streak"] = up * (up.groupby((up != up.shift()).cumsum()).cumcount() + 1)
        F["max_bar_ret_15"] = r1.rolling(15).max()

    def _volume_block(self, F, v, qv, r1):
        F["vol_burst_5_60"] = v.rolling(5).mean() / (v.rolling(60).mean() + EPS)
        F["vol_burst_15_240"] = v.rolling(15).mean() / (v.rolling(240).mean() + EPS)
        F["vol_z_240"] = (v - v.rolling(240).mean()) / (v.rolling(240).std() + EPS)
        F["dollar_vol_60"] = np.log1p(qv.rolling(60).mean())
        F["amihud_60"] = (r1.abs() / np.log1p(qv + 1.0)).rolling(60).mean()

    def _flow_block(self, F, buy, sell, v):
        net = buy - sell
        for w in self.FLOW_WINDOWS:
            F[f"flow_imb_{w}"] = net.rolling(w).sum() / (v.rolling(w).sum() + EPS)   # -1..1
            F[f"buy_ratio_{w}"] = buy.rolling(w).sum() / (v.rolling(w).sum() + EPS)  # 0..1
        F["flow_imb_accel"] = F["flow_imb_5"] - F["flow_imb_60"]
        cvd = net.cumsum()
        F["cvd_slope_30"] = (cvd - cvd.shift(30)) / (v.rolling(30).sum() + EPS)
        F["cvd_slope_120"] = (cvd - cvd.shift(120)) / (v.rolling(120).sum() + EPS)

    def _trades_block(self, F, qv, trd):
        F["trades_z_60"] = (trd - trd.rolling(60).mean()) / (trd.rolling(60).std() + EPS)
        F["trade_intensity"] = trd.rolling(5).mean() / (trd.rolling(60).mean() + EPS)
        F["avg_trade_size_15"] = (qv.rolling(15).sum()) / (trd.rolling(15).sum() + EPS)
        F["avg_trade_size_ratio"] = ((qv.rolling(5).sum() / (trd.rolling(5).sum() + EPS)) /
                                     (qv.rolling(60).sum() / (trd.rolling(60).sum() + EPS) + EPS))

    def _btc_block(self, F, c, r1):
        bc = self.btc["close"].reindex(c.index, method="ffill")
        bbuy = self.btc["tbb"].reindex(c.index, method="ffill")
        bvol = self.btc["volume"].reindex(c.index, method="ffill")
        br1 = bc.pct_change()
        F["btc_ret_15"] = bc / bc.shift(15) - 1.0
        F["btc_ret_60"] = bc / bc.shift(60) - 1.0
        F["btc_rv_60"] = br1.rolling(60).std()
        F["btc_drawdown_240"] = bc / bc.rolling(240).max() - 1.0
        F["btc_flow_imb_60"] = (2 * bbuy - bvol).rolling(60).sum() / (bvol.rolling(60).sum() + EPS)
        F["rel_strength_60"] = self._rets[60] - (bc / bc.shift(60) - 1.0)
        F["beta_btc_120"] = r1.rolling(120).cov(br1) / (br1.rolling(120).var() + EPS)
        F["corr_btc_120"] = r1.rolling(120).corr(br1)

    def _time_block(self, F, idx):
        hr = idx.hour + idx.minute / 60.0
        F["hour_sin"] = np.sin(2 * np.pi * hr / 24)
        F["hour_cos"] = np.cos(2 * np.pi * hr / 24)
        F["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
        F["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)

    # ------------------------------------------------------------------ #
    # Sampling: pull the event (PoA) rows out of a generated frame.
    # ------------------------------------------------------------------ #
    @staticmethod
    def sample_at(frame: pd.DataFrame, event_times) -> pd.DataFrame:
        """Return the feature rows at each event time, using the **last bar at or
        before** the event (`method="ffill"`). Because the frame is already causal,
        this is leak-free — do NOT add an extra shift on top of the rolling columns.

        event_times : ms-epoch ints, ISO strings, or Timestamps. Returns one row
        per event with an `event_time` column.
        """
        idx = pd.Index(event_times)
        # integer/float input is epoch milliseconds (v3 / feature_backtest entry
        # tables); without unit="ms" pandas would read them as nanoseconds -> 1970.
        unit = "ms" if pd.api.types.is_numeric_dtype(idx) else None
        et = pd.to_datetime(idx, utc=True, unit=unit)
        pos = frame.index.get_indexer(et, method="ffill")
        out = frame.iloc[pos].copy()
        out.insert(0, "event_time", et)
        out.index = range(len(out))
        return out

    # ------------------------------------------------------------------ #
    # Event-relative features — kept OUT of the rolling frame on purpose:
    # they need onset context (the start of the move), so igni-bot builds them in
    # the experiment builder, not the frame. Provided here as an explicit, separate
    # step for callers that have onset positions.
    # ------------------------------------------------------------------ #
    @staticmethod
    def event_relative(df: pd.DataFrame, onset_idx: int, poa_idx: int,
                       pre_win: int = 30) -> dict:
        """The 4 event-relative features for one (onset -> PoA) pair.

        ret_since_onset, pullback_from_peak, vol_vs_pre, flow_since_onset
        (verbatim from igni-bot experiment_fut.build_base).
        """
        c = df["close"].to_numpy(); hg = df["high"].to_numpy()
        v = df["volume"].to_numpy(); buy = df["tbb"].to_numpy()
        os_, p = onset_idx, poa_idx
        pre_v = v[max(0, os_ - pre_win):os_].mean()
        bm = v[os_:p + 1].sum()
        return {
            "ret_since_onset": c[p] / c[os_] - 1.0,
            "pullback_from_peak": c[p] / hg[os_:p + 1].max() - 1.0,
            "vol_vs_pre": v[os_:p + 1].mean() / pre_v if pre_v else np.nan,
            "flow_since_onset": (2 * buy[os_:p + 1].sum() - bm) / bm if bm else np.nan,
        }


if __name__ == "__main__":
    # tiny synthetic self-check: shape, leak-safety sanity, btc on/off
    import numpy as _np
    n = 600
    idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    rng = _np.random.default_rng(0)
    close = 100 * _np.exp(_np.cumsum(rng.normal(0, 0.001, n)))
    df = pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999, "close": close,
        "volume": rng.uniform(10, 100, n), "quote_vol": rng.uniform(1e4, 1e5, n),
        "trades": rng.integers(5, 50, n).astype(float),
        "tbb": rng.uniform(5, 50, n),
    }, index=idx)
    btc = df[["close", "tbb", "volume"]].copy()

    fg = FeatureGenerator(btc=btc)
    frame = fg.generate(df)
    print("with btc :", frame.shape[1], "features")
    print("no btc   :", FeatureGenerator().generate(df).shape[1], "features")
    sample = fg.sample_at(frame, [idx[300], idx[500]])
    print("sampled rows:", sample.shape, "| any inf:", _np.isinf(sample.select_dtypes('number')).any().any())

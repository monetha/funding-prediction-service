"""L1 SL-hit classifier — pipeline spec, causal walk-forward evaluation, feature set.

OWNED copy, adapted from funding-trader-bot's position_model.py + position_model_lr.py.
This is the single place that defines:

  * `feature_cols(df)`  — the exact feature set + column order (pinned into feats.json).
  * `make_model()`      — the scaler+L1-logistic pipeline (the chosen, validated model).
  * `walk_forward(df)`  — the honest causal metric: expanding-window MONTHLY walk-forward,
                          AUC on pooled out-of-fold scores. This is the plan's headline
                          `walk_forward_auc` (~0.654 on the validation dataset).

Target: 1 = stopped out early = closed_at < intended_close (funding-grid anchored;
built in dataset.py). Early-but-profitable closes are dropped upstream as ambiguous.

    python model.py [position_dataset.parquet]     # full report incl. walk-forward AUC
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

# sklearn 1.8 deprecated the `penalty` arg in favor of `l1_ratio`; we keep penalty=
# for clarity (the surviving-coef count confirms L1 is honored). Silence the noise.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", message="Inconsistent values: penalty")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

TARGET = "target"

# Non-feature bookkeeping columns carried through the dataset. Defined HERE (next to
# feature_cols, the train/serve parity anchor); dataset.py imports META_COLS from this
# module so the "what is a feature vs metadata" boundary has exactly one definition.
META_COLS = ["position_id", "token", "symbol", "opened_at", "closed_at",
             "funding_period", "funding_rate", "entry_price", "intended_close",
             "margin_min", "realized_pnl", "target"]

# Outcome-derived columns that must never enter X even if a future dataset carries them
# (per-$ return / notional are computed from the trade result -> would leak the label).
LEAK_COLS = ["ret", "notional"]

# Chosen model — strong-L1 logistic. Small C zeroes ~62 of 65 features and keeps a tiny
# interpretable set (hl_range_30, ret_240, max_bar_ret_15, ...), which generalizes
# forward best. Validated causal walk-forward AUC ~0.654 (0.72 on the recent window).
PENALTY = "l1"
C = 0.01


def feature_cols(df: pd.DataFrame) -> list[str]:
    """Ordered feature columns: numeric, minus META, minus outcome-derived leak cols.
    Column order follows the frame's column order (deterministic from featuregen), and
    is what train.py pins into feats.json for exact serve-time reindexing."""
    drop = set(META_COLS) | set(LEAK_COLS)
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def make_model(penalty: str = PENALTY, C: float = C):
    """StandardScaler -> L1 LogisticRegression pipeline.

    The scaler is fit INSIDE the pipeline, so on every `.fit` it re-fits on that train
    slice only — never on held-out rows — leaking no test statistics. class_weight
    balanced handles the ~14-18% SL base rate; liblinear supports the l1 penalty for
    binary targets. random_state pinned for reproducibility."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty=penalty, C=C, class_weight="balanced",
            max_iter=1000, solver="liblinear", random_state=42,
        ),
    )


def event_group(df: pd.DataFrame) -> pd.Series:
    """Funding-event cluster key: opened_at floored to the fp-hour grid. Positions from
    the same funding payment share byte-identical btc_*/time features and market-wide
    correlated targets, so they must never straddle a train/test split."""
    return df.apply(lambda r: r["opened_at"].floor(f"{int(r['funding_period'])}h"), axis=1)


def _prep(df: pd.DataFrame):
    """Sort by open time, return (df, feats, X float32, y int)."""
    df = df.sort_values("opened_at").reset_index(drop=True)
    feats = feature_cols(df)
    X = df[feats].astype("float32")
    y = df[TARGET].astype(int).to_numpy()
    return df, feats, X, y


def walk_forward(df: pd.DataFrame, penalty: str = PENALTY, C: float = C) -> dict:
    """Causal expanding-window MONTHLY walk-forward — the honest 'would this have worked
    live?' metric. For each calendar month m (after the first), train on ALL positions
    opened in months < m and score the positions opened in month m; AUC is computed once
    on the pooled out-of-fold scores.

    Returns {auc, n_scored, n_months, base_rate, oof} where oof is a Series (index-aligned
    to the sorted df) of out-of-fold p_sl (NaN for the seed month / skipped months)."""
    df, feats, X, y = _prep(df)
    month = df["opened_at"].dt.to_period("M")
    oof = np.full(len(df), np.nan)
    for m in sorted(month.unique())[1:]:
        tr = (month < m).to_numpy()
        te = (month == m).to_numpy()
        # need enough history and both classes present in train to fit a usable model
        if te.sum() == 0 or tr.sum() < 50 or y[tr].sum() == 0 or y[tr].sum() == tr.sum():
            continue
        oof[te] = make_model(penalty, C).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    scored = ~np.isnan(oof)
    auc = (roc_auc_score(y[scored], oof[scored])
           if scored.sum() and len(np.unique(y[scored])) > 1 else float("nan"))
    return {
        "auc": float(auc),
        "n_scored": int(scored.sum()),
        "n_months": int(month.nunique()),
        "base_rate": float(y.mean()),
        "oof": pd.Series(oof, index=df.index),
    }


def holdout_auc(df: pd.DataFrame, penalty: str = PENALTY, C: float = C) -> float:
    """Time-ordered 80/20 holdout AUC, cut snapped to a funding-event edge so no event
    straddles train/test (would leak correlated twins). Scaler fits on the train slice
    only. Secondary metric alongside walk_forward()."""
    df, feats, X, y = _prep(df)
    grp = event_group(df)
    cut = int(len(df) * 0.8)
    while cut < len(df) and grp.iloc[cut] == grp.iloc[cut - 1]:
        cut += 1
    if cut >= len(df) or len(np.unique(y[cut:])) < 2:
        return float("nan")
    m = make_model(penalty, C).fit(X.iloc[:cut], y[:cut])
    return float(roc_auc_score(y[cut:], m.predict_proba(X.iloc[cut:])[:, 1]))


def grouped_cv_auc(df: pd.DataFrame, penalty: str = PENALTY, C: float = C,
                   n_splits: int = 5) -> float:
    """5-fold GroupKFold AUC, whole funding events held out together (no twin leak)."""
    df, feats, X, y = _prep(df)
    grp = event_group(df)
    p = cross_val_predict(make_model(penalty, C), X, y, groups=grp,
                          cv=GroupKFold(n_splits=n_splits),
                          method="predict_proba", n_jobs=-1)[:, 1]
    return float(roc_auc_score(y, p))


def fit(df: pd.DataFrame, penalty: str = PENALTY, C: float = C):
    """Fit the pipeline on the FULL frame (production model). Returns (pipeline, feats)."""
    df, feats, X, y = _prep(df)
    return make_model(penalty, C).fit(X, y), feats


def surviving_coefs(pipeline, feats: list[str]) -> pd.Series:
    """Non-zero standardized coefficients of a fitted pipeline, ordered by |magnitude|
    descending (sign = direction of SL risk). Logged as an artifact at train time."""
    lr = pipeline.named_steps["logisticregression"]
    coef = pd.Series(lr.coef_[0], index=feats)
    kept = coef[coef != 0]
    return kept.reindex(kept.abs().sort_values(ascending=False).index)


def report(df: pd.DataFrame) -> dict:
    """Print the full L1 evaluation: walk-forward (headline) + holdout + grouped CV +
    surviving coefficients. Returns the metric dict."""
    _, feats, _, y = _prep(df)
    wf = walk_forward(df)
    hold = holdout_auc(df)
    cv = grouped_cv_auc(df)
    pipe, _ = fit(df)
    kept = surviving_coefs(pipe, feats)

    print("\n" + "=" * 64)
    print("L1 logistic SL-hit classifier  (target=1 -> closed early / SL)")
    print("=" * 64)
    print(f"positions: {len(df):,}   features: {len(feats)}   base rate (SL): {y.mean():.1%}")
    print(f"\n[causal walk-forward, expanding monthly]  scored={wf['n_scored']} "
          f"over {wf['n_months']} months")
    print(f"  ROC AUC = {wf['auc']:.4f}   <-- headline walk_forward_auc")
    print(f"\n[time-ordered holdout, cluster-safe]   ROC AUC = {hold:.4f}")
    print(f"[5-fold GroupKFold CV, by funding event]  ROC AUC = {cv:.4f}")
    print(f"\nsurviving features ({len(kept)} of {len(feats)}; sign = direction of SL risk):")
    for name, val in kept.items():
        print(f"  {name:<28} {val:+.4f}")
    return {"walk_forward_auc": wf["auc"], "holdout_auc": hold, "cv_auc": cv,
            "n_nonzero_coef": int(len(kept)), "base_rate": float(y.mean()), "features": feats}


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    path = sys.argv[1] if len(sys.argv) > 1 else "position_dataset.parquet"
    report(pd.read_parquet(path))

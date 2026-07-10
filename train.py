"""Train the L1 SL-hit classifier, log+register to MLflow, gate, and promote.

Pipeline:
  1. build the labeled dataset (dataset.build) — CLOSED positions -> features -> target.
  2. fit the L1 scaler+logistic pipeline on the FULL frame (the production model).
  3. compute the honest metrics: walk_forward_auc (headline) + holdout + grouped-CV +
     n_nonzero_coef + base rate.
  4. log params / metrics / artifacts (ordered feats.json = training column order;
     surviving-coef table) and register the model as `sl_classifier`.
  5. GATE + PROMOTE: move the `production` alias to the new version iff
     walk_forward_auc >= max(0.58, prod_auc - 0.02). First version promoted
     unconditionally. Otherwise keep the current production model.

    python train.py                    # build + train + log + gate
    python train.py --dataset ds.parquet   # reuse a prebuilt dataset

Requires the MLflow tracking server (docker compose up mlflow) at SLP_MLFLOW_URI.
"""
from __future__ import annotations

import argparse
import logging
import os

import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

import dataset as dataset_mod
import model as M
from data_sources import DEFAULT_FUNDING_DB

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRACKING_URI = os.environ.get("SLP_MLFLOW_URI", "http://127.0.0.1:5000")
EXPERIMENT = os.environ.get("SLP_EXPERIMENT", "sl_classifier")
REGISTERED_MODEL = "sl_classifier"
PROD_ALIAS = "production"

# Promotion gate: never promote below ABS_FLOOR, and never regress more than SLACK
# below the incumbent production model's walk-forward AUC.
ABS_FLOOR = 0.58
SLACK = 0.02


def _prod_walk_forward_auc(client: MlflowClient) -> float | None:
    """walk_forward_auc of the current @production version, or None if none exists."""
    try:
        mv = client.get_model_version_by_alias(REGISTERED_MODEL, PROD_ALIAS)
    except Exception:
        return None
    run = client.get_run(mv.run_id)
    return run.data.metrics.get("walk_forward_auc")


def _gate(new_auc: float, prod_auc: float | None) -> tuple[bool, str]:
    """Decide promotion. Returns (promote?, human-readable reason)."""
    if prod_auc is None:
        return True, "first version — promoted unconditionally"
    threshold = max(ABS_FLOOR, prod_auc - SLACK)
    if new_auc >= threshold:
        return True, (f"{new_auc:.4f} >= threshold {threshold:.4f} "
                      f"(max({ABS_FLOOR}, prod {prod_auc:.4f} - {SLACK}))")
    return False, (f"{new_auc:.4f} < threshold {threshold:.4f} "
                   f"(max({ABS_FLOOR}, prod {prod_auc:.4f} - {SLACK})) — keeping prod")


def train(ds: pd.DataFrame, source: str | None = None) -> dict:
    """Fit + evaluate + log + register + gate. Returns a summary dict.

    `source` is the provenance URI of `ds` (parquet path, dataset.csv, funding db);
    it only labels the logged dataset — the digest is hashed from `ds` itself.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    feats = M.feature_cols(ds)
    logger.info("dataset: %d rows, %d features, base rate %.1f%%",
                len(ds), len(feats), ds[M.TARGET].mean() * 100)

    # --- honest metrics (computed on the labeled frame) ---
    wf = M.walk_forward(ds)
    holdout = M.holdout_auc(ds)
    cv = M.grouped_cv_auc(ds)
    pipeline, feats = M.fit(ds)                      # production model: full-frame fit
    kept = M.surviving_coefs(pipeline, feats)
    logger.info("walk_forward_auc=%.4f  holdout=%.4f  cv=%.4f  nonzero_coef=%d",
                wf["auc"], holdout, cv, len(kept))

    Xf = ds.sort_values("opened_at").reset_index(drop=True)[feats].astype("float32")

    with mlflow.start_run() as run:
        # the labeled frame the production model is fit on — digest changes whenever
        # positions are added/relabeled, even if `source` is a stable path
        mlflow.log_input(
            mlflow.data.from_pandas(ds, source=source, targets=M.TARGET,
                                    name="sl_positions"),
            context="train",
        )
        mlflow.log_params({
            "penalty": M.PENALTY, "C": M.C,
            "feature_lag_min": dataset_mod.FEATURE_LAG_MIN,
            "close_lead_min": dataset_mod.CLOSE_LEAD_MIN,
            "solver": "liblinear", "class_weight": "balanced",
            "n_features": len(feats),
        })
        mlflow.log_metrics({
            "walk_forward_auc": wf["auc"],
            "holdout_auc": holdout,
            "cv_auc": cv,
            "n_nonzero_coef": float(len(kept)),
            "base_rate": wf["base_rate"],
            "n_positions": float(len(ds)),
            "wf_n_scored": float(wf["n_scored"]),
        })
        # ordered feature list — the exact training column order serving must reindex to
        mlflow.log_dict(list(feats), "feats.json")
        # surviving-coef table (sign = direction of SL risk)
        mlflow.log_dict({k: float(v) for k, v in kept.items()}, "surviving_coefs.json")

        signature = infer_signature(Xf, pipeline.predict_proba(Xf))
        mlflow.sklearn.log_model(
            pipeline, artifact_path="model",
            registered_model_name=REGISTERED_MODEL,
            signature=signature,
            input_example=Xf.head(),
        )
        run_id = run.info.run_id
        logger.info("logged run %s", run_id)

    # the version just registered from this run
    new_version = max(
        (mv for mv in client.search_model_versions(f"name='{REGISTERED_MODEL}'")
         if mv.run_id == run_id),
        key=lambda mv: int(mv.version),
    ).version

    # --- gate + promote ---
    prod_auc = _prod_walk_forward_auc(client)
    promote, reason = _gate(wf["auc"], prod_auc)
    logger.info("GATE: %s", reason)
    if promote:
        client.set_registered_model_alias(REGISTERED_MODEL, PROD_ALIAS, new_version)
        logger.info("PROMOTED %s v%s -> @%s", REGISTERED_MODEL, new_version, PROD_ALIAS)
    else:
        logger.info("NOT promoted; @%s stays at its current version", PROD_ALIAS)

    return {
        "run_id": run_id, "version": new_version,
        "walk_forward_auc": wf["auc"], "holdout_auc": holdout, "cv_auc": cv,
        "n_nonzero_coef": len(kept), "promoted": promote, "gate_reason": reason,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", help="prebuilt parquet; if omitted, build from source")
    args = ap.parse_args()

    if args.dataset:
        ds, source = pd.read_parquet(args.dataset), args.dataset
    else:
        ds, source = dataset_mod.build(), str(DEFAULT_FUNDING_DB)
    if ds.empty:
        logger.error("empty dataset — aborting")
        return
    summary = train(ds, source=source)
    print("\n" + "=" * 64)
    print(f"run {summary['run_id']}  ->  {REGISTERED_MODEL} v{summary['version']}")
    print(f"walk_forward_auc={summary['walk_forward_auc']:.4f}  "
          f"holdout={summary['holdout_auc']:.4f}  cv={summary['cv_auc']:.4f}  "
          f"nonzero_coef={summary['n_nonzero_coef']}")
    print(f"promoted={summary['promoted']}  ({summary['gate_reason']})")
    print("=" * 64)


if __name__ == "__main__":
    main()

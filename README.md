# SL-Probability Prediction Service

A standalone, self-contained service that predicts **`p_sl`** — the probability that a
Binance funding position is *stopped out early* (closed before its intended, funding-grid
anchored close). The funding bot sends `{symbol, exchange, save}` and gets back a raw
`p_sl`; **the bot owns all sizing.** This service is prediction-only.

Model: **L1 (sparse) logistic regression** over 65 causal features. Validated causal
walk-forward AUC **0.654** (0.72 on the recent window) — the best-generalizing of the
models tried.

> **Scope:** Binance only (the model is Binance-trained; other exchanges are rejected).
> No bot / sizing code lives here. The service returns raw `p_sl`, nothing else.

---

## Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │              docker compose stack             │
                       │                                               │
   funding bot ──HTTP──┼──▶ predictor (FastAPI :8100)                  │
   {symbol,exchange}   │        │  load @production once at startup    │
        p_sl ◀─────────┼────────┤  ~25s BTC refresher (in-memory)      │
                       │        │  live 1m klines ◀── Binance fapi     │
                       │        │  append feature vector ──▶ feature-store (vol) │
                       │        ▼                                       │
                       │     mlflow (:5000)  sqlite + serve-artifacts   │
                       │        ▲                                       │
                       │     retrain (one-shot, cron) ─┐               │
                       │        build → label log → union → gate →      │
                       │        promote @production → POST /reload      │
                       └──────────────────────────────────────────────┘
   read-only mounts (retrain only): kline cache, funding_bot.db
```

- **Model loaded once at startup** into memory; requests never touch MLflow. `/reload`
  hot-swaps after a retrain.
- **Shared BTC frame** refreshed by a ~25s background task; requests read it from memory
  with **zero BTC fetches on the hot path**. Token klines get a short per-symbol TTL to
  absorb bursts.
- **Production predictions persist** their 65-feature vector + `p_sl` to a sqlite feature
  store for later labeling / retraining. Only requests with `"save": true` are stored, so
  probes, tests and manual curls never pollute the training log.
- **Fail-closed:** missing/stale BTC, too little history, or any NaN feature → error;
  never a silent fall back to the 57-feature (no-BTC) input.

---

## Repo layout

| File | Purpose |
|---|---|
| `featuregen.py` | **Owned** copy of the 65-feature causal engine — the single source of train/serve parity (both sides import THIS file). |
| `model.py` | L1 pipeline spec, `feature_cols` (+ `META_COLS`), and `walk_forward()` (the honest metric). |
| `data_sources.py` | Read raw inputs: `KlineCache` (parquet cache) + `load_closed_positions` (labels from `funding_bot.db`). |
| `dataset.py` | Build the labeled set: CLOSED positions → features @ `open − 1min` → SL target. |
| `train.py` | Build → fit → log/register to MLflow → AUC gate → promote `@production`. |
| `dataset_csv.py` | The incremental training set (`data/dataset.csv`). Owns **all** feature sourcing: cached csv → feature store → Binance REST. |
| `retrain_monthly.py` | Schedule-agnostic CLI: `dataset_csv.sync()` + retrain + `/reload` + drift audit. |
| `retrain_cron.sh` | Monthly cron wrapper → `docker compose run --rm retrain`. |
| `serving/binance_klines.py` | Raw 1m klines fetch (token + BTC) from Binance public REST; CCXT→Binance symbol; drop partial bar. |
| `serving/feature_store.py` | Append-only sqlite log of `{meta, 65 feats, p_sl}`; logging never fails a response. |
| `serving/app.py` | FastAPI: startup load, BTC refresher, `POST /predict`, `GET /health`, `POST /reload`. |
| `tests/test_parity.py` | Offline `p_sl` == service `p_sl` for the same bar (~1e-6). |
| `docker/mlflow/Dockerfile` | MLflow tracking+registry server (sqlite backend). |
| `docker/service/Dockerfile` | Train+serve image (shared) — `python:3.12-slim` + `libgomp1`. |
| `docker-compose.yml` | `mlflow`, `predictor`, `retrain` (profile-gated). |
| `requirements.txt` | Pinned train+serve environment (see below). |

---

## Quickstart

```bash
# 1. tracking server + prediction service
docker compose up -d              # starts mlflow (:5000) + predictor (:8100)

# 2. train the first model (required before predictor can serve)
docker compose run --rm retrain   # builds dataset, logs+registers, promotes @production, reloads

# 3. predict (add "save":true only when the bot is really opening on this signal)
curl -s -X POST localhost:8100/predict \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"SOL/USDT:USDT","exchange":"binance"}'
# -> {"p_sl":0.38,"model_version":"3","event_time":"2026-07-08T16:42:00Z","saved":false}

curl -s localhost:8100/health
# -> {"status":"ok","model_loaded":true,"model_version":"3","n_features":65,"btc_age_sec":6.1,...}
```

> The `predictor` container will not serve until a model is promoted to `@production`.
> Run `retrain` (or `train.py`) once first.

---

## Request / response contract

`POST /predict` — body `{ "symbol": "SOL/USDT:USDT", "exchange": "binance", "save": false }`

`save` (optional, default `false`) marks the call as a real production prediction: only
then is the feature vector written to the feature store and picked up by the next retrain.
The response echoes `saved`, which is `true` only if the row actually landed — a store
failure is still non-fatal and returns `saved: false` with a `200`.

| Status | Body | When |
|---|---|---|
| `200` | `{ "p_sl": 0.14, "model_version": "3", "event_time": "…Z", "saved": false }` | success |
| `400` | `{ "detail": "unsupported exchange" }` | `exchange != binance` |
| `422` | `{ "detail": "insufficient history" }` | too few bars / NaN features (warmup) |
| `503` | `{ "detail": "btc frame unavailable/stale" }` | BTC missing or older than `SLP_BTC_MAX_AGE_SEC` (retryable) |
| `502` | `{ "detail": "kline fetch failed: …" }` | upstream Binance error (incl. unknown/delisted symbol) |

`GET /health` → model loaded flag, version, feature count, BTC age, predictions logged.
`POST /reload` → reloads `@production` (model + `feats.json` + version) in place.

### Prediction log

Every `/predict` call — served or rejected, `save` or not — appends one JSON line to
`logs/predictions.log` (bind-mounted from the container's `/logs`, set by `SLP_LOG_DIR`).
This is the request audit trail; the feature store holds only the `save: true` rows.

```
{"ts":"…Z","symbol":"SOL/USDT:USDT","exchange":"binance","save":true,"status":200,
 "p_sl":0.421126,"model_version":"3","event_time":"…Z","saved":true,"latency_ms":17.3}
{"ts":"…Z","symbol":"SOL/USDT:USDT","exchange":"bybit","save":true,"status":400,
 "detail":"unsupported exchange","latency_ms":0.0}
```

Rotates at `SLP_LOG_MAX_BYTES` (50 MB) × `SLP_LOG_BACKUPS` (5). A missing or unwritable
log dir disables the log with a warning — it never fails a request. Unset `SLP_LOG_DIR`
to turn it off. The container's stdout (uvicorn access lines + the 15s healthcheck) is
capped separately by the compose `logging:` block at 10 MB × 3.

> **Deviations from the original plan (intentional):** BTC missing/stale returns **503**
> (retryable/transient) rather than 422, so the bot knows to retry; an unknown/delisted
> symbol currently surfaces as **502** (could be refined to a 400 "invalid symbol").

---

## Feature engine & the SL target

- **Input schema (per 1m bar):** `open, high, low, close, volume, quote_vol, trades, tbb`
  on a UTC index. `tbb` = taker-buy base volume (only in Binance *raw* klines, not ccxt
  `fetch_ohlcv`). BTC needs only `close, tbb, volume` → drives the 8 `btc_*` features.
  **65 features total** with BTC (57 without — never used in serving).
- **Serving sampling:** fetch `limit=300` 1m klines, **drop the in-progress last bar**,
  sample at `now.floor("min") − 1min` (matches training `FEATURE_LAG_MIN=1`), then reindex
  to the pinned `feats.json` column order before `predict_proba`.
- **Target:** `1` = stopped out early = `closed_at < intended_close`, where
  `intended_close = floor(open, fp·h) + fp·h − 30min` (funding-grid anchored,
  `CLOSE_LEAD_MIN=30`). Early-but-**profitable** closes are dropped as ambiguous.

## Model & promotion gate

```python
make_pipeline(
    StandardScaler(),
    LogisticRegression(penalty="l1", C=0.01, class_weight="balanced",
                       solver="liblinear", max_iter=1000, random_state=42))
```

- **Metric — `walk_forward_auc`:** expanding-window **monthly** walk-forward (train on
  months `< m`, score month `m`; AUC on pooled out-of-fold scores). Reproduced at
  **0.6536** on the validation set. Secondary: time-ordered holdout + GroupKFold (grouped
  by funding event).
- **Gate:** promote the new version to `@production` iff
  `walk_forward_auc ≥ max(0.58, prod_auc − 0.02)`. First version promoted unconditionally.
- Strong L1 zeroes ~60 of 65 features, keeping a small set (`hl_range_30`, `ret_240`,
  `max_bar_ret_15`, …) — logged as the `surviving_coefs.json` artifact each run.

---

## Environment / pinning

Train and serve **share one environment** (one image, one `requirements.txt`) so the
MLflow-logged model round-trips byte-for-byte and features are identical on both sides.

| Pin | Why |
|---|---|
| `scikit-learn==1.8.0` | the env that validated AUC 0.654; `model.py` matches its `penalty=` behavior |
| `pandas==2.3.3` | mlflow 2.22.0 caps `pandas<3`, and serving imports mlflow + computes features in one process |
| `numpy==2.2.6`, `scipy==1.16.1` | consistent with the above |
| `mlflow==2.22.0` | matches the tracking/registry server image |
| `pyarrow==19.0.1` | mlflow 2.22.0 caps `pyarrow<20` |
| `fastapi`, `uvicorn`, `aiohttp`, `requests`, `xgboost`, `pytest` | serving / benchmarks / tests |

**Feature-engine parity under pandas 2 is verified:** rebuilding the dataset under the
pinned env reproduces the validation set exactly (2384 rows, 14.3% base rate, AUC 0.6536),
with max feature diff vs the original **1.04e-6** (concentrated in `ret_kurt_60`; not a
surviving feature). Since prod runs pandas 2 on both train and serve, real train/serve
parity is exact.

### Configuration (env vars)

| Var | Default | Used by |
|---|---|---|
| `SLP_MLFLOW_URI` | `http://127.0.0.1:5000` | train, serve, retrain |
| `SLP_SERVICE_URI` | `http://127.0.0.1:8100` | retrain (`/reload`), parity test |
| `SLP_KLINE_CACHE` | `/root/funding-trader-bot/feature_mining/data` | dataset/retrain |
| `SLP_FUNDING_DB` | `/root/funding-trader-bot/db/funding_bot.db` | dataset/retrain (labels) |
| `SLP_FEATURE_STORE_DB` | `serving/feature_store.db` | serve/retrain |
| `SLP_BINANCE_FAPI` | `https://fapi.binance.com` | serve |
| `SLP_BTC_REFRESH_SEC` / `SLP_BTC_MAX_AGE_SEC` | `25` / `90` | serve |
| `SLP_TOKEN_TTL_SEC` | `8` | serve |

---

## Retraining & scheduling

`retrain_monthly.py` is a **schedule-agnostic, idempotent CLI**:
1. `dataset_csv.sync()` → the training set (see below);
2. retrain → log → **gate** → promote `@production`;
3. `POST /reload` on the predictor;
4. **drift audit** — logged vs recomputed features for the same bar (audit only; the
   feature log is *not* a training source).

### `data/dataset.csv` — the incremental training set

The igni kline cache is a **static snapshot** (it ends at its last cached bar and nothing
refreshes it). `dataset.build()` silently skips positions with no kline coverage, so retrain
was slowly going blind to recent history — 110 closed positions, growing ~13/day, were
already invisible. `dataset_csv.py` fixes that by caching each position's features once and
backfilling only what's missing, with this precedence:

| # | source | `source` column | when |
|---|---|---|---|
| 1 | `data/dataset.csv` | `cache` / `rest` / `store` | already cached — never recomputed |
| 2 | `feature_store.db` | `store` | the bot served this bar with `save: true` |
| 3 | Binance REST | `rest` | recompute from historical klines |

```bash
python dataset_csv.py --rebuild   # bootstrap, or after ANY featuregen.py edit
python dataset_csv.py --sync      # backfill missing positions (what retrain calls)
```

Two operational rules this creates:

- **`data/` is irreplaceable — back it up.** If `dataset.csv` is lost while the kline cache
  is stale, recent rows can only be rebuilt by refetching from Binance.
- **Editing `featuregen.py` means `--rebuild`.** Cached rows freeze the feature definitions
  that produced them. `dataset.csv.meta.json` stores a hash of `featuregen.py`; a mismatch
  **aborts the retrain** rather than silently mixing two feature definitions in one training
  set. `dataset.csv.skip.json` records positions that can never be fetched (delisted tokens)
  so they aren't retried every run.

`data/` is mounted as a **directory**, never as a single file: `save_atomic()` uses
`os.replace`, which swaps the inode, and a file bind-mount would pin the stale one.

MLflow has **no built-in scheduler or webhooks** (open-source) — the trigger is external:

```
# root crontab — 03:00 UTC on the 1st of each month, flock-guarded
0 3 1 * * /usr/bin/flock -n /tmp/slp_retrain.lock /root/funding-prediction-service/retrain_cron.sh
```

`retrain_cron.sh` runs `docker compose run --rm retrain` and appends to `logs/retrain.log`.

---

## Operations runbook

```bash
docker compose ps                          # stack status
docker compose logs -f predictor           # tail service logs
docker compose restart predictor           # restart serving
curl -X POST localhost:8100/reload         # hot-swap to current @production
docker compose run --rm retrain            # manual retrain now
docker compose build predictor && docker compose up -d predictor   # deploy code change

# inspect the feature store (inside the shared volume)
docker compose exec predictor \
  python -c "import sqlite3;print(sqlite3.connect('/data/feature_store.db').execute('select symbol,p_sl,model_version from prediction_log order by id desc limit 5').fetchall())"

# parity test (needs mlflow + service running)
.venv/bin/python -m pytest tests/test_parity.py -v
```

**Verified behavior:** 20 concurrent predicts complete in ~0.6s (sub-second) with zero
BTC-frame fetches on the hot path; `test_parity.py` asserts service `p_sl` == offline
recompute to <1e-6.

### Local (non-Docker) dev

A host `.venv/` mirrors the pinned image for quick iteration:
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python train.py
.venv/bin/uvicorn serving.app:app --port 8100
```

---

## Out of scope

- Any `funding_bot.py` / sizing change (the bot merely calls `/predict`).
- Returning a size factor — raw `p_sl` only.
- Non-Binance exchanges (rejected until separately trained).

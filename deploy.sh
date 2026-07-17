#!/usr/bin/env bash
# Deploy the SL-Probability Prediction Service: bring up the docker stack and install
# the monthly retrain cron job. Safe to re-run — every step is idempotent.
#
#   ./deploy.sh              # build, start stack, install cron
#   ./deploy.sh --no-cron    # stack only (e.g. a host that shouldn't retrain)
#   ./deploy.sh --no-build   # skip image build (faster; config-only redeploy)
#
# Bootstrap order matters: serving/app.py loads `sl_classifier@production` in its
# lifespan, so a predictor started before any model exists crash-loops under
# `restart: unless-stopped`. On a fresh host we mint the first model before starting it.
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${REPO}/logs"

# cron runs with a minimal PATH; docker/compose/flock live in these dirs.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

CRON_LOCK="/tmp/slp_retrain.lock"
CRON_LINE="0 3 1 * * /usr/bin/flock -n ${CRON_LOCK} ${REPO}/retrain_cron.sh"
CRON_COMMENT="# SL-predictor: monthly retrain (03:00 UTC on the 1st)"

# retrain reads these read-only; only needed on the bootstrap path.
KLINE_CACHE="/root/funding-trader-bot/feature_mining/data"
FUNDING_DB="/root/funding-trader-bot/db/funding_bot.db"

DO_CRON=1
DO_BUILD=1
for arg in "$@"; do
  case "$arg" in
    --no-cron)  DO_CRON=0 ;;
    --no-build) DO_BUILD=0 ;;
    -h|--help)  sed -n '2,9p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

log()  { printf '\n[deploy] %s\n' "$*"; }
fail() { printf '\n[deploy] ERROR: %s\n' "$*" >&2; exit 1; }

cd "${REPO}"

# --------------------------------------------------------------------------- #
# 0. preflight
# --------------------------------------------------------------------------- #
log "preflight"
command -v docker >/dev/null || fail "docker not found on PATH"
docker compose version >/dev/null 2>&1 || fail "'docker compose' (v2) not available"
[[ -f "${REPO}/docker-compose.yml" ]] || fail "docker-compose.yml not found in ${REPO}"
[[ -x "${REPO}/retrain_cron.sh" ]] || fail "retrain_cron.sh missing or not executable"
if (( DO_CRON )); then
  command -v flock >/dev/null || fail "flock not found (util-linux); needed by the cron line"
fi
mkdir -p "${LOG_DIR}"

# --------------------------------------------------------------------------- #
# 1. build the shared train+serve image
# --------------------------------------------------------------------------- #
if (( DO_BUILD )); then
  log "building images"
  docker compose build
else
  log "skipping build (--no-build)"
fi

# --------------------------------------------------------------------------- #
# 2. mlflow first — the registry must be healthy before we can query the alias
# --------------------------------------------------------------------------- #
log "starting mlflow (waiting for healthy)"
docker compose up -d --wait mlflow

# --------------------------------------------------------------------------- #
# 3. bootstrap: mint the first model if @production doesn't exist yet
# --------------------------------------------------------------------------- #
# Run the check inside the service image so it uses the pinned mlflow client and the
# compose-network URI from the compose env — never the host .venv.
has_production_model() {
  docker compose run --rm --no-deps -T predictor python -c '
import sys
import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri(__import__("os").environ["SLP_MLFLOW_URI"])
try:
    MlflowClient().get_model_version_by_alias("sl_classifier", "production")
except Exception:
    sys.exit(1)
' >/dev/null 2>&1
}

if has_production_model; then
  log "@production model already exists — skipping bootstrap retrain"
else
  log "no @production model — bootstrapping with a first retrain"
  # This branch only fires on a fresh host, which is exactly where the read-only
  # training inputs are most likely to be missing. Fail loudly rather than half-deploy.
  [[ -d "${KLINE_CACHE}" ]] || fail "kline cache not found: ${KLINE_CACHE} (needed to train)"
  [[ -f "${FUNDING_DB}"  ]] || fail "funding db not found: ${FUNDING_DB} (needed for labels)"
  docker compose run --rm retrain \
    || fail "bootstrap retrain failed — predictor not started (it would crash-loop)"
fi

# --------------------------------------------------------------------------- #
# 4. predictor — safe to start now that a model is promoted
# --------------------------------------------------------------------------- #
log "starting predictor (waiting for healthy)"
docker compose up -d --wait predictor

# --------------------------------------------------------------------------- #
# 5. cron — install the monthly retrain line exactly once
# --------------------------------------------------------------------------- #
if (( DO_CRON )); then
  # `crontab -l` exits non-zero on an empty crontab; don't let that abort the script.
  current="$(crontab -l 2>/dev/null || true)"
  if grep -Fq 'retrain_cron.sh' <<<"${current}"; then
    log "cron entry already present — leaving crontab untouched"
  else
    log "installing monthly retrain cron entry"
    # Rewrite the whole crontab (that's the only interface), preserving existing lines.
    printf '%s\n%s\n%s\n' "${current}" "${CRON_COMMENT}" "${CRON_LINE}" \
      | sed '/^$/d' | crontab -
  fi
else
  log "skipping cron install (--no-cron)"
fi

# --------------------------------------------------------------------------- #
# 6. report
# --------------------------------------------------------------------------- #
log "stack status"
docker compose ps

log "service health"
docker compose exec -T predictor python -c '
import json, urllib.request
print(json.dumps(json.load(urllib.request.urlopen("http://localhost:8100/health")), indent=2))
' || fail "predictor is up but /health did not answer"

if (( DO_CRON )); then
  log "installed cron entries"
  crontab -l | grep -F 'retrain_cron.sh' || true
fi

log "deploy complete"

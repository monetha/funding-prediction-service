#!/usr/bin/env bash
# Monthly retrain wrapper for cron. Runs the retrain as a one-shot compose container
# (same image as the predictor). cron has a minimal environment, so set PATH explicitly
# and use absolute paths. flock (in the crontab line) prevents overlapping runs.
set -euo pipefail

REPO="/root/funding-prediction-service"
LOG_DIR="${REPO}/logs"
mkdir -p "${LOG_DIR}"

# docker/compose live in /usr/bin; ensure they're on PATH under cron.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

cd "${REPO}"
{
  echo "===== retrain start $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  # --rm: remove the one-shot container after it exits. depends_on brings up mlflow.
  docker compose run --rm retrain
  echo "===== retrain done  $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
} >> "${LOG_DIR}/retrain.log" 2>&1

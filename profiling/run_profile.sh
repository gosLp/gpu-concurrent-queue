#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# User-facing environment variables
# ============================================================
PROFILE_QUEUES="${PROFILE_QUEUES:-gwfq glfq wfq sfq}"
PROFILE_WORKLOAD="${PROFILE_WORKLOAD:-tb}"
PROFILE_FIFO_MODE="${PROFILE_FIFO_MODE:-0}"
PROFILE_OUT_ROOT="${PROFILE_OUT_ROOT:-results/profiling}"
PROFILE_BIN_ROOT="${PROFILE_BIN_ROOT:-out/profile}"
PROFILE_EXTRA_MAKE_FLAGS="${PROFILE_EXTRA_MAKE_FLAGS:-}"

CSV_PROCESSOR="${CSV_PROCESSOR:-profiling/csv_processor.py}"

# Optional manual override:
#   GPU_FAMILY=mi210 make profile-tb
#   GPU_FAMILY=mi300a make profile-tb
GPU_FAMILY="${GPU_FAMILY:-auto}"

# ============================================================
# Helpers
# ============================================================

detect_gpu_family() {
  if [[ "${GPU_FAMILY}" != "auto" ]]; then
    echo "${GPU_FAMILY}"
    return
  fi

  local info
  info="$(rocminfo 2>/dev/null | tr '\n' ' ' || true)"

  if echo "$info" | grep -Eqi 'MI300|gfx942'; then
    echo "mi300a"
  elif echo "$info" | grep -Eqi 'MI210|gfx90a'; then
    echo "mi210"
  else
    echo "unknown"
  fi
}

metric_tag() {
  basename "$1" .in
}

latest_csv() {
  local dir="$1"
  find "$dir" -type f -path "*/pmc_*/*" -name 'results_*.csv' -printf '%T@ %p\n' \
    | sort -nr | head -n1 | cut -d' ' -f2-
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required file missing: $path" >&2
    exit 2
  fi
}

# ============================================================
# GPU-specific metric selection
# ============================================================

GPU="$(detect_gpu_family)"

case "$GPU" in
  mi210)
    METRIC_FILES=(
      "profiling/metrics/common/stalls.in"
      "profiling/metrics/common/ea_atomic.in"
      "profiling/metrics/mi210/occ.in"
    )
    ;;
  mi300a)
    METRIC_FILES=(
      "profiling/metrics/common/stalls.in"
      "profiling/metrics/common/ea_atomic.in"
      "profiling/metrics/mi300a/occa.in"
    )
    ;;
  *)
    echo "WARNING: Could not identify GPU family from rocminfo."
    echo "         Falling back to common metrics only."
    echo "         Override with GPU_FAMILY=mi210 or GPU_FAMILY=mi300a."
    METRIC_FILES=(
      "profiling/metrics/common/stalls.in"
      "profiling/metrics/common/ea_atomic.in"
    )
    ;;
esac

echo "[PROFILE] GPU family      : $GPU"
echo "[PROFILE] workload        : $PROFILE_WORKLOAD"
echo "[PROFILE] queues          : $PROFILE_QUEUES"
echo "[PROFILE] fifo mode       : $PROFILE_FIFO_MODE"
echo "[PROFILE] output root     : $PROFILE_OUT_ROOT"
echo "[PROFILE] binary root     : $PROFILE_BIN_ROOT"
echo

# Validate metrics early.
for metric in "${METRIC_FILES[@]}"; do
  require_file "$metric"
done

mkdir -p "$PROFILE_OUT_ROOT"
mkdir -p "$PROFILE_BIN_ROOT"

# ============================================================
# Main throughput profiling path
# ============================================================

if [[ "$PROFILE_WORKLOAD" != "tb" ]]; then
  echo "ERROR: currently supported PROFILE_WORKLOAD is only 'tb'." >&2
  exit 2
fi

for queue in $PROFILE_QUEUES; do
  QUEUE_OUT="$PROFILE_OUT_ROOT/$GPU/tb/$queue/fifo${PROFILE_FIFO_MODE}"
  BIN_DIR="$PROFILE_BIN_ROOT/$GPU/tb"
  BIN="$BIN_DIR/tb_${queue}_fifo${PROFILE_FIFO_MODE}"

  mkdir -p "$QUEUE_OUT/logs"
  mkdir -p "$BIN_DIR"

  echo "[PROFILE] Building queue=$queue"
  make --no-print-directory tb-build \
    QUEUE="$queue" \
    GPU_FAMILY="$GPU" \
    TB_FIFO_MODE="$PROFILE_FIFO_MODE" \
    TB_BUILD_DIR="$BIN_DIR" \
    TB_CSV_FILE="$QUEUE_OUT/throughput_rows.csv" \
    $PROFILE_EXTRA_MAKE_FLAGS

  if [[ ! -x "$BIN" ]]; then
    echo "ERROR: expected binary not found or not executable: $BIN" >&2
    exit 2
  fi

  for metric_in in "${METRIC_FILES[@]}"; do
    tag="$(metric_tag "$metric_in")"
    RAW_DIR="$QUEUE_OUT/raw/$tag"
    FINAL_CSV="$QUEUE_OUT/${GPU}_${queue}_fifo${PROFILE_FIFO_MODE}_${tag}.csv"
    LOG_FILE="$QUEUE_OUT/logs/${GPU}_${queue}_fifo${PROFILE_FIFO_MODE}_${tag}.log"

    mkdir -p "$RAW_DIR"

    echo "[PROFILE] queue=$queue metric=$tag"
    echo "          binary : $BIN"
    echo "          output : $FINAL_CSV"

    rocprofv2 --plugin file -i "$metric_in" -d "$RAW_DIR" "$BIN" \
      2>&1 | tee "$LOG_FILE"

    RAW_CSV="$(latest_csv "$RAW_DIR")"

    if [[ -z "${RAW_CSV:-}" ]]; then
      echo "ERROR: rocprofv2 produced no CSV under $RAW_DIR" >&2
      exit 1
    fi

    cp -f "$RAW_CSV" "$FINAL_CSV"

    if [[ -n "$CSV_PROCESSOR" && -x "$CSV_PROCESSOR" ]]; then
      PROC_OUT="${FINAL_CSV%.csv}_processed.csv"
      python3 "$CSV_PROCESSOR" "$FINAL_CSV" "$PROC_OUT"
      echo "          processed: $PROC_OUT"
    fi

    echo
  done
done

echo "[PROFILE] Completed all profiling runs."
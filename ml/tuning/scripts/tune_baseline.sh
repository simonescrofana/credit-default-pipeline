#!/bin/bash
# tune_baseline.sh
#
# Manual grid search over the baseline (LogisticRegression) hyperparameters:
# for each value of C, overwrites the default in ml/models/baseline.py, runs
# training (CV + final fit), extracts the aggregated CV metrics from the
# log, and appends them to a CSV.
#
# max_iter stays fixed: it's a convergence cap, not a model hyperparameter
# that changes prediction quality (same reasoning already applied to
# XGBoost's early_stopping_rounds).
#
# run_training.py's __main__ block only trains whichever model is hardcoded
# there (currently "xgboost_model"), so this script also temporarily patches
# that line to select "baseline" — restored on exit exactly like the model
# file below.
#
# Usage (must be run from the project root, since the paths below are
# relative):
#   ./ml/tuning/scripts/tune_baseline.sh
# (make it executable once with: chmod +x ml/tuning/scripts/tune_baseline.sh)

set -euo pipefail

MODEL_FILE="ml/models/baseline.py"
RUNNER_FILE="ml/run_training.py"
RESULTS="ml/tuning/results/baseline_results.csv"
LOG_DIR="ml/tuning/logs/baseline"

mkdir -p "$LOG_DIR" "$(dirname "$RESULTS")"

# Back up both files, always restored on exit (success, error, or Ctrl+C).
cp "$MODEL_FILE" "${MODEL_FILE}.bak"
cp "$RUNNER_FILE" "${RUNNER_FILE}.bak"
trap 'cp "${MODEL_FILE}.bak" "$MODEL_FILE" && rm -f "${MODEL_FILE}.bak"; \
      cp "${RUNNER_FILE}.bak" "$RUNNER_FILE" && rm -f "${RUNNER_FILE}.bak"' EXIT

# Select "baseline" as the model to train, whatever the current selection is.
sed -i \
    -e 's/^\( *\)main(models_to_train=\[.*\]).*/\1main(models_to_train=["baseline"])/' \
    -e 's/^\( *\)main()\s*#.*/\1main(models_to_train=["baseline"])/' \
    "$RUNNER_FILE"

# CSV header, only if the file doesn't exist yet (lets you resume an
# interrupted run without losing previous results).
if [ ! -f "$RESULTS" ]; then
    echo "C,auc_roc,auc_pr,precision,recall,f1" > "$RESULTS"
fi

C_VALUES=(0.01 0.1 1.0 10.0 100.0)

TOTAL=${#C_VALUES[@]}
COUNT=0

for c in "${C_VALUES[@]}"; do
    COUNT=$((COUNT + 1))

    # ^ anchors the start of the line: replaces whatever value is currently
    # there, not just the original default.
    sed -i "s/^DEFAULT_C = .*/DEFAULT_C = $c/" "$MODEL_FILE"

    echo "[$COUNT/$TOTAL] C=$c"

    LOG_FILE="$LOG_DIR/run_C${c}.log"

    # Training logs go to stdout/stderr (StreamHandler); save the full
    # output and also grep it below.
    uv run python -m ml.run_training > "$LOG_FILE" 2>&1

    # Line of interest, produced by trainer.py:
    #   Aggregated cross-validation metrics: {'auc_roc': 0.71, 'auc_pr': 0.65, ...}
    #
    # NOTE: setup_logging() registers both a plain StreamHandler and
    # Logfire's handler, and Logfire *also* echoes every line to
    # stdout in its own compact format. The same message therefore
    # appears twice in the captured log (once per handler). -m 1
    # takes only the first occurrence, so the values below are
    # extracted once, not twice.
    METRICS_LINE=$(grep -m 1 "Aggregated cross-validation metrics:" "$LOG_FILE" || true)

    if [ -z "$METRICS_LINE" ]; then
        echo "  WARNING: no metrics line found, check $LOG_FILE"
        echo "$c,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$RESULTS"
        continue
    fi

    # Extract the Python dict after the colon, then each numeric value tied
    # to its key via grep -oP (Perl regex).
    DICT_PART="${METRICS_LINE#*Aggregated cross-validation metrics: }"

    extract_metric() {
        local key="$1"
        echo "$DICT_PART" | grep -oP "'${key}': \K[0-9.eE+-]+" || echo "NA"
    }

    AUC_ROC=$(extract_metric "auc_roc")
    AUC_PR=$(extract_metric "auc_pr")
    PRECISION=$(extract_metric "precision")
    RECALL=$(extract_metric "recall")
    F1=$(extract_metric "f1")

    echo "  auc_pr=$AUC_PR auc_roc=$AUC_ROC f1=$F1"
    echo "$c,$AUC_ROC,$AUC_PR,$PRECISION,$RECALL,$F1" >> "$RESULTS"

done

echo ""
echo "Done. Results in $RESULTS, full logs in $LOG_DIR/"
echo "Top rows sorted by descending auc_pr:"
{ head -1 "$RESULTS"; tail -n +2 "$RESULTS" | sort -t, -k3 -rn; } | column -t -s,

#!/bin/bash
# tune_xgb.sh
#
# Manual grid search over XGBoost hyperparameters: for each combination of
# (max_depth, learning_rate, n_estimators), overwrites the defaults in
# ml/models/xgboost_model.py, runs training (CV + final fit), extracts the
# aggregated CV metrics from the log, and appends them to a CSV.
#
# early_stopping_rounds stays fixed (it's not a model hyperparameter, see
# discussion: it only affects CV, and is disabled for the final fit).
#
# Usage (must be run from the project root, since FILE below is a relative
# path):
#   ./ml/tuning/scripts/fine_tune_xgb.sh
# (make it executable once with: chmod +x ml/tuning/scripts/fine_tune_xgb.sh)

set -euo pipefail

FILE="ml/models/xgboost_model.py"
RUNNER_FILE="ml/run_training.py"
RESULTS="ml/tuning/results/xgboost_fine_results.csv"
LOG_DIR="ml/tuning/logs/xgboost_fine_tune"

mkdir -p "$LOG_DIR" "$(dirname "$RESULTS")"

# Back up both files, always restored on exit (success, error, or Ctrl+C),
# so the repo is never left dirty with the last combination tried.
cp "$FILE" "${FILE}.bak"
cp "$RUNNER_FILE" "${RUNNER_FILE}.bak"
trap 'cp "${FILE}.bak" "$FILE" && rm -f "${FILE}.bak"; \
      cp "${RUNNER_FILE}.bak" "$RUNNER_FILE" && rm -f "${RUNNER_FILE}.bak"' EXIT

# Select "xgboost_model" as the model to train, whatever the current
# selection is (explicit, rather than relying on it already being the
# hardcoded default in run_training.py).
sed -i \
    -e 's/^\( *\)main(models_to_train=\[.*\]).*/\1main(models_to_train=["xgboost_model"])/' \
    -e 's/^\( *\)main()\s*#.*/\1main(models_to_train=["xgboost_model"])/' \
    "$RUNNER_FILE"

# CSV header, only if the file doesn't exist yet (lets you resume an
# interrupted run without losing previous results).
if [ ! -f "$RESULTS" ]; then
    echo "max_depth,learning_rate,n_estimators,auc_roc,auc_pr,precision,recall,f1" > "$RESULTS"
fi

MAX_DEPTH_VALUES=(6 8 10 12)
LEARNING_RATE_VALUES=(0.05 0.1 0.15 0.2)
N_ESTIMATORS_VALUES=(200)

TOTAL=$((${#MAX_DEPTH_VALUES[@]} * ${#LEARNING_RATE_VALUES[@]} * ${#N_ESTIMATORS_VALUES[@]}))
COUNT=0

for depth in "${MAX_DEPTH_VALUES[@]}"; do
    for lr in "${LEARNING_RATE_VALUES[@]}"; do
        for n_est in "${N_ESTIMATORS_VALUES[@]}"; do
            COUNT=$((COUNT + 1))

            sed -i "s/^DEFAULT_MAX_DEPTH = .*/DEFAULT_MAX_DEPTH = $depth/" "$FILE"
            sed -i "s/^DEFAULT_LEARNING_RATE_XGB = .*/DEFAULT_LEARNING_RATE_XGB = $lr/" "$FILE"
            sed -i "s/^DEFAULT_N_ESTIMATORS = .*/DEFAULT_N_ESTIMATORS = $n_est/" "$FILE"

            echo "[$COUNT/$TOTAL] max_depth=$depth learning_rate=$lr n_estimators=$n_est"

            LOG_FILE="$LOG_DIR/run_depth${depth}_lr${lr}_nest${n_est}.log"

            # Training logs go to stdout/stderr (StreamHandler); save the
            # full output and also grep it below.
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
                echo "$depth,$lr,$n_est,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$RESULTS"
                continue
            fi

            # Extract the Python dict after the colon, then each numeric
            # value tied to its key via grep -oP (Perl regex).
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
            echo "$depth,$lr,$n_est,$AUC_ROC,$AUC_PR,$PRECISION,$RECALL,$F1" >> "$RESULTS"

        done
    done
done

echo ""
echo "Done. Results in $RESULTS, full logs in $LOG_DIR/"
echo "Top rows sorted by descending auc_pr:"
{ head -1 "$RESULTS"; tail -n +2 "$RESULTS" | sort -t, -k5 -rn; } | column -t -s,

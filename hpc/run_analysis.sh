#!/usr/bin/env bash
set -euo pipefail
umask 077
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${1:-python}"
cd "$PROJECT_DIR"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MPLBACKEND=Agg
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PROJECT_DIR/results/.matplotlib"
mkdir -p results/logs "$MPLCONFIGDIR"
CONFIG_PATH="${SMSC_CONFIG:-configs/analysis.json}"
"$PYTHON_BIN" -B -m unittest discover -s tests -v > results/logs/unit_tests.log 2>&1
arguments=(scripts/reproduce.py --config "$CONFIG_PATH" --explanations)
if [[ -n "${SMSC_CASES:-}" ]]; then
  arguments+=(--cases "$SMSC_CASES")
fi
"$PYTHON_BIN" -B "${arguments[@]}"
while IFS= read -r split_seed; do
  "$PYTHON_BIN" -B scripts/model_selection.py --config "$CONFIG_PATH" --project . --output results/model_selection --seed "$split_seed" --threads 2 > "results/logs/cv_seed_${split_seed}.log" 2>&1
done < <("$PYTHON_BIN" -c 'import json,sys; print("\n".join(map(str, json.load(open(sys.argv[1]))["seeds"])))' "$CONFIG_PATH")
"$PYTHON_BIN" -B scripts/model_selection.py --config "$CONFIG_PATH" --output results/model_selection --summarize
echo "Analysis completed. Outputs are in results/."

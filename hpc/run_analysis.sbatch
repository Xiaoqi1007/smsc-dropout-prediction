#!/usr/bin/env bash
#SBATCH --job-name=smsc-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=smsc-analysis-%j.out
#SBATCH --error=smsc-analysis-%j.err

set -euo pipefail
umask 077
PROJECT_DIR="${1:-$SLURM_SUBMIT_DIR}"
PYTHON_BIN="${2:-python}"
cd "$PROJECT_DIR"
exec bash hpc/run_analysis.sh "$PYTHON_BIN"

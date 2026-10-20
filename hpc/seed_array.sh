#!/bin/bash
# Cluster array job: one seed per task, all tasks writing into the same stamped
# result set (per-seed files and concurrency-safe stamping make this safe; a
# resubmitted array skips seeds that already finished).
#
# Usage (from the repository root):
#   sbatch --array=0-349 hpc/seed_array.sh run_final_experiment
#   sbatch --array=0-34  hpc/seed_array.sh run_calibration_experiment
#   sbatch --array=0-19 --cpus-per-task=8 hpc/seed_array.sh run_record_confidence --m 10 25 50 100 --threads 8
#
# Adapt the resource lines and the module setup to the cluster (check the site
# documentation, e.g. how to select a cluster and which Python module exists).
#SBATCH --job-name=rtds-seed
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --output=hpc/logs/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
# module load Python/3.12   # ADAPT: load the same Python the .venv was created with
source .venv/bin/activate

RUNNER=$1; shift
SEED=${SLURM_ARRAY_TASK_ID}
python -u "experiments/${RUNNER}.py" --seed-start "${SEED}" --seed-end "$((SEED + 1))" --workers 1 "$@"

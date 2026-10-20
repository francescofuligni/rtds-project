#!/bin/bash
# Cluster array job: one seed per task, all tasks writing into the same stamped
# result set results/$RTDS_RUN_TIMESTAMP/<experiment>/ (per-seed files and
# concurrency-safe stamping make this safe; resubmitting the array with the same
# timestamp skips seeds that already finished).
#
# Usage (from the repository root):
#   export RTDS_RUN_TIMESTAMP=$(python3 experiments/output_paths.py)
#   sbatch --array=0-349 hpc/seed_array.sh run_final_experiment
#   sbatch --array=0-34  hpc/seed_array.sh run_calibration_experiment
#   sbatch --array=0-19  hpc/seed_array.sh run_record_confidence --m 10 25 50 100
#
# Every task passes --workers 1, so all tasks stamp the same protocol.
# Adapt the resource lines and the module setup to the cluster (check the site
# documentation, e.g. how to select a cluster and which Python module exists).
#SBATCH --job-name=rtds-seed
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --output=hpc/logs/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
: "${RTDS_RUN_TIMESTAMP:?export RTDS_RUN_TIMESTAMP before sbatch so all tasks share one result set}"
export RTDS_RUN_TIMESTAMP
# module load Python/3.12   # ADAPT: load the same Python the .venv was created with
source .venv/bin/activate

RUNNER=$1; shift
SEED=${SLURM_ARRAY_TASK_ID}
python -u "experiments/${RUNNER}.py" --seed-start "${SEED}" --n-seeds 1 --workers 1 "$@"

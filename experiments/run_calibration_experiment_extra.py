"""One-off supplementary run: adds 15 more seeds (20..34) to the
calibration_experiment.csv sample, to grow the held-out calibration
test set (currently the identified bottleneck for Brier/ECE precision,
not agreement-scale resolution -- see results.tex). Writes to a
separate file; merged into calibration_experiment.csv afterward.

Usage: .venv/bin/python experiments/run_calibration_experiment_extra.py
"""

from __future__ import annotations

import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from run_calibration_experiment import N_WORKERS, run_seed

SEED_START = 20
SEED_END = 35  # exclusive -> seeds 20..34, 15 new seeds

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    t_start = time.time()

    all_rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(run_seed, seed): seed for seed in range(SEED_START, SEED_END)}
        for future in as_completed(futures):
            seed = futures[future]
            try:
                all_rows.extend(future.result())
            except Exception as e:
                print(f"[seed {seed}] FAILED: {e}", flush=True)

    out_path = RESULTS_DIR / "calibration_experiment_part2.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows to {out_path} in {time.time() - t_start:.1f}s "
          f"({N_WORKERS} parallel workers, {SEED_END - SEED_START} seeds)")


if __name__ == "__main__":
    main()

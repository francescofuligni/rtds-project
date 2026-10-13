"""One-off supplementary run: adds 150 more seeds (200..349) to the main
final_experiment.csv sample, to increase statistical power for the per-m
Holm-Bonferroni tests (condition A in particular is significant pooled
but not at any single m with the original 200 seeds). Writes to a
separate file; merge_extra_seeds.py concatenates it with the original.

Usage: .venv/bin/python experiments/run_final_experiment_extra.py
"""

from __future__ import annotations

import csv
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from run_final_experiment import N_WORKERS, run_seed

SEED_START = 200
SEED_END = 350  # exclusive -> seeds 200..349, 150 new seeds

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    t_start = time.time()

    all_rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(run_seed, seed): seed for seed in range(SEED_START, SEED_END)
        }
        for future in as_completed(futures):
            seed = futures[future]
            try:
                all_rows.extend(future.result())
            except Exception:
                logging.getLogger(__name__).exception("[seed %s] FAILED", seed)

    out_path = RESULTS_DIR / "final_experiment_part2.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(
        f"\nWrote {len(all_rows)} rows to {out_path} in {time.time() - t_start:.1f}s "
        f"({N_WORKERS} parallel workers, {SEED_END - SEED_START} seeds)"
    )


if __name__ == "__main__":
    main()

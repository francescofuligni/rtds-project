"""Run the supplementary main batch for seeds 200 through 349."""

from run_final_experiment import main

if __name__ == "__main__":
    main(
        [
            "--seed-start",
            "200",
            "--seed-end",
            "350",
            "--output",
            "results/final_experiment_part2.csv",
        ]
    )

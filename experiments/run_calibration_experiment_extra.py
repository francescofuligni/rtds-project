"""Run the supplementary diagnostic batch for seeds 20 through 34."""

from run_calibration_experiment import main

if __name__ == "__main__":
    main(
        [
            "--seed-start",
            "20",
            "--seed-end",
            "35",
        ],
        output_name="calibration_experiment_part2.csv",
    )

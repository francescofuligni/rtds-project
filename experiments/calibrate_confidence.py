"""Turn the solver-agreement confidence proxy into an actual calibrated
probability, and validate the calibration on held-out seeds.

Section results-confidence showed agreement rate *correlates* with
correctness, but never produced a usable probability, nor validated
anything on data the calibration mapping hadn't already seen. This
fits an isotonic regression (agreement -> P(correct)) per m on a
training split of seeds, and evaluates Brier score + expected
calibration error (ECE) against raw agreement-as-probability on a
disjoint set of held-out seeds.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = ROOT / "results" / "calibration_experiment.csv"
FIGURES_DIR = ROOT / "report" / "figures"

TEST_SEED_FRACTION = 0.3
SPLIT_SEED = 0
N_ECE_BINS = 10
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 1


def expected_calibration_error(prob: np.ndarray, correct: np.ndarray, n_bins: int = N_ECE_BINS) -> float:
    """Standard ECE: bin by predicted probability, weight each bin's
    |accuracy - mean predicted prob| by its share of the data."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = prob[mask].mean()
        ece += (mask.sum() / len(prob)) * abs(bin_acc - bin_conf)
    return ece


def brier_score(prob: np.ndarray, correct: np.ndarray) -> float:
    return float(np.mean((prob - correct) ** 2))


def bootstrap_ci(test_df: pd.DataFrame, iso: IsotonicRegression, n_boot: int = N_BOOTSTRAP) -> dict:
    """Seed-level block bootstrap (not row-level) since rows within a seed
    share the same pilot sample and Rashomon set and are not independent.
    Resamples test seeds with replacement, pools their rows, and recomputes
    all four metrics; returns 2.5/97.5 percentile CIs."""
    seeds = test_df["seed"].unique()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    by_seed = {s: g for s, g in test_df.groupby("seed")}
    results = {"raw_brier": [], "calibrated_brier": [], "raw_ece": [], "calibrated_ece": []}
    for _ in range(n_boot):
        sample_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        boot = pd.concat([by_seed[s] for s in sample_seeds], ignore_index=True)
        agreement = boot["agreement"].to_numpy()
        correct = boot["correct"].to_numpy().astype(float)
        calibrated = iso.predict(agreement)
        results["raw_brier"].append(brier_score(agreement, correct))
        results["calibrated_brier"].append(brier_score(calibrated, correct))
        results["raw_ece"].append(expected_calibration_error(agreement, correct))
        results["calibrated_ece"].append(expected_calibration_error(calibrated, correct))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in results.items()}


def main():
    df = pd.read_csv(RESULTS_CSV)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    seeds = sorted(df["seed"].unique())
    rng = np.random.default_rng(SPLIT_SEED)
    shuffled = rng.permutation(seeds)
    n_test = max(1, int(round(len(seeds) * TEST_SEED_FRACTION)))
    test_seeds = set(shuffled[:n_test].tolist())
    train_seeds = set(shuffled[n_test:].tolist())
    print(f"Train seeds ({len(train_seeds)}): {sorted(train_seeds)}")
    print(f"Test seeds ({len(test_seeds)}): {sorted(test_seeds)}")

    m_values = sorted(df["m"].unique())
    fig, axes = plt.subplots(1, len(m_values), figsize=(5.5 * len(m_values), 4.2), sharey=True)
    if len(m_values) == 1:
        axes = [axes]

    summary_rows = []
    per_m_models = {}
    for ax, m in zip(axes, m_values):
        sub = df[df["m"] == m]
        train = sub[sub["seed"].isin(train_seeds)]
        test = sub[sub["seed"].isin(test_seeds)]

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(train["agreement"].to_numpy(), train["correct"].to_numpy().astype(float))
        per_m_models[m] = (iso, test)

        test_agreement = test["agreement"].to_numpy()
        test_correct = test["correct"].to_numpy().astype(float)
        calibrated = iso.predict(test_agreement)

        raw_brier = brier_score(test_agreement, test_correct)
        cal_brier = brier_score(calibrated, test_correct)
        raw_ece = expected_calibration_error(test_agreement, test_correct)
        cal_ece = expected_calibration_error(calibrated, test_correct)
        ci = bootstrap_ci(test, iso)

        summary_rows.append({
            "m": m, "n_train": len(train), "n_test": len(test),
            "raw_brier": raw_brier, "calibrated_brier": cal_brier,
            "raw_ece": raw_ece, "calibrated_ece": cal_ece,
            "raw_brier_ci": ci["raw_brier"], "calibrated_brier_ci": ci["calibrated_brier"],
            "raw_ece_ci": ci["raw_ece"], "calibrated_ece_ci": ci["calibrated_ece"],
        })

        # reliability diagram on the TEST split only: raw agreement vs. isotonic-calibrated
        raw_tab = test.groupby("agreement")["correct"].agg(["mean", "count"]).reset_index()
        ax.scatter(raw_tab["agreement"], raw_tab["mean"],
                   s=(raw_tab["count"] / raw_tab["count"].max() * 300 + 20),
                   alpha=0.6, color="#d95f02", edgecolor="black", linewidth=0.5, label="raw agreement (held-out)")

        grid = np.linspace(test_agreement.min(), test_agreement.max(), 100)
        ax.plot(grid, iso.predict(grid), color="#1b9e77", linewidth=2, label="isotonic-calibrated")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1, label="perfect calibration")

        ax.set_xlim(0.4, 1.05)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Agreement rate (raw) / calibrated probability")
        ax.set_title(f"m = {m}")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Empirical accuracy (held-out test seeds)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Isotonic-calibrated confidence vs. raw agreement, evaluated on held-out seeds", fontsize=11)
    fig.tight_layout()

    out_path = FIGURES_DIR / "calibration_isotonic.pdf"
    fig.savefig(out_path)
    print(f"\nWrote {out_path}")

    print("\nCalibration quality on held-out seeds (lower is better for both metrics):")
    print(pd.DataFrame(summary_rows)[["m", "n_train", "n_test", "raw_brier", "calibrated_brier",
                                       "raw_ece", "calibrated_ece"]].round(4).to_string(index=False))

    print("\n95% bootstrap CIs (seed-level block bootstrap, 1000 resamples of the held-out test seeds):")
    for row in summary_rows:
        print(f"  m={row['m']}:")
        for metric in ["raw_brier", "calibrated_brier", "raw_ece", "calibrated_ece"]:
            lo, hi = row[f"{metric}_ci"]
            print(f"    {metric:<18} point={row[metric]:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")

    print("\nPooled (m-agnostic) vs. per-m isotonic calibration, evaluated on each m's held-out test set:")
    pooled_train = df[df["seed"].isin(train_seeds)]
    pooled_iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    pooled_iso.fit(pooled_train["agreement"].to_numpy(), pooled_train["correct"].to_numpy().astype(float))
    for m in m_values:
        _, test = per_m_models[m]
        test_agreement = test["agreement"].to_numpy()
        test_correct = test["correct"].to_numpy().astype(float)
        pooled_calibrated = pooled_iso.predict(test_agreement)
        pooled_brier = brier_score(pooled_calibrated, test_correct)
        pooled_ece = expected_calibration_error(pooled_calibrated, test_correct)
        per_m_row = next(r for r in summary_rows if r["m"] == m)
        print(f"  m={m}: per-m Brier={per_m_row['calibrated_brier']:.4f} ECE={per_m_row['calibrated_ece']:.4f}"
              f"  |  pooled Brier={pooled_brier:.4f} ECE={pooled_ece:.4f}")


if __name__ == "__main__":
    main()

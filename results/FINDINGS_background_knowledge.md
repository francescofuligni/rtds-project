# Findings: background knowledge and the number of released trees

Status 2026-10-21. The main paired experiment (handoff step 1), run with two
background-attribute pairs on the same seeds and releases.

## Setup

- n=100 private rows, target T=sex, random trees from the first 1,000 of the
  Rashomon set, geometric grid m = 1, 2, 3, 5, 7, 10, 14, 21, 30, 44, 64, 94,
  137, 200 (K=15, m_max=200), 180 s per solve, 4 CP-SAT workers.
- Conditions: None, A, B, A+B (the attacker knows the column(s) for every
  row). A = marital-status in both sets; B = education in one, age in the
  other. Seeds 0-99 in both, so each seed has the same private sample and
  the same released trees in both sets.
- Error = share of wrong sex cells in the reconstructed table. Reference:
  uniform guess 0.503, marginal guess 0.445.

## Results

**1. More trees lower the error.** Without background knowledge the error
falls from 0.346 at m=1 to 0.253 at m=200. Paired by seed against m=1, the
drop is significant from m=10 (-2.3 points, p=0.002) and reaches -9 points
at m=137-200 (p<1e-10), in every condition and both pairs.

**2. Background knowledge helps a little.** Mean gain over None, pooled over
all m (Wilcoxon paired by seed and m, about 1,377 cells):

| knowledge | marital + education set | marital + age set |
|---|---|---|
| A (marital-status) | 0.6 points, p=0.018 | 0.6 points, p=0.006 |
| B | education: 0.6 points, p=0.0002 | age: 0.1 points, p=0.42 |
| A+B | 1.3 points, p=1.5e-13 | 0.9 points, p=2e-6 |

This confirms the earlier first-m run (A+B: 2.0 points pooled, p=1e-34), at a
smaller size: random trees carry more information themselves, so less is left
for background knowledge to add. Per m, A+B with education beats None by 0.8-2.2 points (p<0.05 at 9 of 14
m); with age only at m=200 (2.2 points, p=0.002). No single per-m test
survives Holm-Bonferroni correction (0 of 42 in each set). Education is
the more useful B: the difference in gain is significant at a few m only
(B at m=2, 7, 137; A+B at m=1, 7), so the pairs differ in degree, not in
kind.

**3. The joint-split mechanism is not supported.** A tree's share of
root-to-leaf paths that split on both sex and A (or B) does not predict
the gain from knowing A (or B): r = -0.01 to -0.03, all p>0.2, in both
sets.

## Caveats

- Solved share: at least 99% of solves found a table in every condition
  and m (2 of 5,516 in the education set timed out). From m=30 on a few
  cells are missing because that seed's Rashomon set has fewer than m
  trees (6 seeds have fewer than 200); m=200 has 94 seeds.
- Solver nondeterminism: the None and A conditions are the same problem in
  both sets, yet 15% (None) and 5% (A) of cells return a table with a
  different error, rising to about 65% at m>=137. Several CP-SAT workers
  race, and which of several feasible tables comes back varies. The noise
  is unbiased (mean difference 0.000, sd 0.020 per cell for None) but of
  the same order as the background-knowledge gains, which is one reason
  per-m tests are weak.
- The education set was resumed once (commits 3ea503a and a5a4ffe); the
  changes between them do not touch this experiment.
- The age set's last 21 seeds ran as two parallel processes (seeds 79-89
  and 90-99) with the same protocol.

## Result sets

| directory | contents |
|---|---|
| `results/20261020_195325/final_experiment/` | A=marital-status, B=education, 100 seeds |
| `results/20261020_195325_01/final_experiment/` | A=marital-status, B=age, 100 seeds |
| `results/comparisons/background_pairs_education_vs_age/` | paired comparison of the two sets |

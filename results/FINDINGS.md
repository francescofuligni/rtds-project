# Findings: attacker-side, per-record confidence

Status 2026-10-21. Numbers below come from the stamped result sets listed at
the end; each set's `summary_analyze_record_confidence.txt` has the full tables.

## Question

The reconstruction attack has a *global* error (share of wrong rows). The goal
is a *local* error: a confidence for every reconstructed record that the
attacker can compute without ground truth, and that is calibrated out of the
box (no fitting) -- among rows stated at 80%, about 80% should be right.

## Setup

- 100 private rows from UCI Adult (13 binary columns, the curated schema, plus
  the income label), a Rashomon set (TreeFARMS, regularisation 0.01, bound
  0.1) built on them, and `m` released trees.
- The attacker also has a public sample of 2,000 *other* people from the same
  population (the strong version of the public-data assumption; a weaker
  version, published per-attribute percentages only, is untested).
- Each attacker outputs a 100-row table (per income label, as many rows as the
  release says carry it) with a confidence per row. A row "record r, copy k"
  is correct if the private table really contains at least k copies of r
  (all 13 columns and the label must match).
- 20 seeds per setting (a seed = a fresh private sample and Rashomon set).

## Attackers

| name | idea |
|---|---|
| `per_tree_presence` | one reconstruction per tree; score = share of trees whose table contains the record |
| `per_tree_absolute` | same reconstructions; score = total copies across them (a ranking, not a probability) |
| `per_tree_occurrence` | same; per copy k, share of trees with at least k copies |
| `seed_only` | 20 joint reconstructions (all trees) differing only in solver seed |
| `prior_guided` | 20 joint reconstructions, each the release-consistent table closest to a random table drawn from the public sample; score = share of samples containing the claim |
| `prior_only` | no trees: P(at least k copies of r) from the public sample's record frequencies |

## Results

**1. Counting reconstructions does not work, for a structural reason.** A single
tree constrains a few columns; CP-SAT fills every unconstrained cell with the
same default (0, i.e. "male" for sex), and consecutive trees are near-copies,
so per-tree (and seed-only) reconstructions agree with each other because
they share solver defaults, not because the data forces them. With random
trees at m=10: per-tree presence is right on 7% of rows at a stated 50%,
the absolute count ranks worse than chance (AUROC 0.31), seed-only is right
on 34% at a stated 86%. The old ~30% global error on sex was largely "the
solver says male".

**2. Which trees are released matters a lot.** TreeFARMS decodes tree indices
like an odometer, so the first m trees are near-copies. Official runs, same
20 seeds, paired:

| m | first m trees | m random trees | difference |
|---|---|---|---|
| 10 | 60.8% (overconfident by 4.0 points) | 69.2% (2.3) | +8.4 points, p=0.0002 |
| 25 | 61.7% (3.9) | 73.9% (1.6) | +12.2 points, p=0.00009 |

With the first m trees, 25 trees tell the attacker barely more than 10;
with random trees accuracy keeps rising. All results below use random
selection; earlier experiments that used the first m trees understate
leakage.

**3. Prior-guided sampling is calibrated out of the box and beats public
statistics.** Official run, random trees, 20 seeds:

| m | prior-guided accuracy | stated confidence - accuracy | ECE | AUROC | public statistics accuracy |
|---|---|---|---|---|---|
| 10 | 69.2% | +2.3 | 0.029 | 0.80 | 58.5% (AUROC 0.73) |
| 25 | 73.9% | +1.6 | 0.028 | 0.82 | 58.5% |
| 50 | 77.8% | +0.7 | 0.025 | 0.83 | 58.5% |

Against public statistics: +10.7 points at m=10 (19/20 seeds better,
p=0.0001), +15.4 at m=25 (20/20, p=0.00009), +19.3 at m=50 (20/20,
p=0.00009); better ranking and Brier score at every m (p<0.001). Its
overconfidence shrinks as more trees are released (+2.3, +1.6, +0.7
points). Its per-seed calibration error (|mean confidence - accuracy|) is
indistinguishable from the public-statistics baseline at m=10 and 25
(p>=0.3) and smaller at m=50 (15/20 seeds, p=0.02); the baseline is itself
near-perfectly calibrated.

**4. The leak is concentrated in records the attacker can vouch for.** Share of
its table an attacker can keep at >= 0.9 confidence (and how many of those
rows are right):

| m | prior-guided | public statistics |
|---|---|---|
| 10 | 31% (95% correct) | 10% (92%) |
| 25 | 39% (95% correct) | 10% (92%) |
| 50 | 45% (98% correct) | 10% (92%) |

The released trees triple to quadruple the number of records an attacker
can name with ~95% reliability (all differences p<0.0001); the table as a
whole gains less. Rows stated at confidence 1.0 (every sample agrees)
are 98.1% correct at m=10 (19 such rows per seed), 99.6% at m=25 (27) and
99.7% at m=50 (34; 2 wrong out of 673).
This is what a per-record confidence shows and a global error hides.

**5. Attempted fixes for the small remaining overconfidence did not help**
(ablation, m=10, 10 seeds): scoring with an independent half of the samples
(against the winner's curse), random tie-breaking among equally close tables,
and doubling the samples all leave accuracy, calibration and AUROC unchanged
within noise (paired p>=0.13). Plain prior-guided sampling is used.

**6. Weaker public data: the leak stays, the calibration loosens.** Same 20
seeds, m=10, random trees, 6 solver threads, prior-guided sampling and the
no-trees baseline under the same public knowledge:

| what the attacker knows | prior-guided accuracy | stated - accuracy | public statistics accuracy | trees add | vouched at >= 0.9 (correct) |
|---|---|---|---|---|---|
| 2,000 other people's records (reference) | 69.4% | +2.1 | 58.5% | +10.9 | 31% (95%) |
| 200 other people's records | 65.1% | +8.6 | 48.3% | +16.9 | 33% (89%) |
| per-attribute shares within each income class (from 2,000 people) | 64.5% | +5.3 | 50.6% | +13.9 | 28% (93%) |

The reference rerun reproduces the official m=10 result (69.2%, +2.3).
With less public knowledge both attackers get worse, the no-trees baseline
more so, so the released trees add more (all prior-guided vs. baseline
differences p<0.0001). But the out-of-the-box calibration loosens:
per-seed calibration error 3.2 -> 8.6 points with 200 records (p=0.00004)
and 3.2 -> 5.6 points with shares only (p=0.015). With few public records
the random tables repeat the same people, so samples agree too often.

**7. Carrying the public data's own uncertainty into the confidence
restores calibration.** Instead of drawing random tables from the public
records as they are, each prior-guided sample first draws a record
distribution from a Dirichlet posterior over all possible records (public
counts + 0.1 per record), then the rows from it; the no-trees baseline
becomes the matching beta-binomial (`--public-prior records_dirichlet`).
With 200 public records (m=10, same 20 seeds):

| | prior-guided accuracy | stated - accuracy | ECE | vouched at >= 0.9 (correct) | public statistics: stated - accuracy |
|---|---|---|---|---|---|
| plain records | 65.1% | +8.6 | 0.087 | 33% (89%) | +7.3 |
| Dirichlet | 65.2% | +1.6 | 0.025 | 24% (95%) | +1.9 |

Accuracy and ranking are unchanged (p>=0.8); only the confidences become
honest again (per-seed calibration error 8.6 -> 3.0 points, p=0.00003).
Whether the Dirichlet prior costs anything with 2,000 records is still
running (13 of 20 seeds done).

## Caveats

- The strong public-data assumption drives the calibration: free cells follow
  the public sample. With a 10x smaller sample or per-attribute shares only,
  the confidences turn 5-9 points overconfident (result 6); a shifted
  public sample (different population) is untested.
- The public-statistics baseline's smoothing (alpha=0.1) was chosen by looking
  at its calibration against the private tables (alpha=1 made it
  underconfident). That tuning favours the baseline, so it is conservative for
  the comparison.
- The attack is given each row's income label; at the record level this is
  equivalent to knowing the class totals, which the released leaf counts
  reveal anyway.
- With 13 binary columns there are only 640 possible records, so a sizeable
  share of rows can be named correctly from population statistics alone
  (~58%). The leak is the difference to that baseline.
- Multi-threaded CP-SAT is not bit-for-bit reproducible; seeds fix the private
  sample, the trees and the public draws, not the solver's search order.

## Result sets

| directory | contents |
|---|---|
| `results/record_confidence/random/` | official run: random trees, all standard attackers, m = 10, 25, 50 (100 to follow) |
| `results/record_confidence/first/` | official run: first m trees (TreeFARMS order), m = 10, 25; compared with the random set in `results/record_confidence/summary_analyze_record_confidence.txt` |
| `results/record_confidence_variants/random/` | ablation of prior-guided variants, m=10 |
| `results/20261020_202208/record_confidence/` | public data: 2,000 records (reference rerun), m=10, prior-guided + public statistics, 6 solver threads |
| `results/20261020_210941/record_confidence/` | public data: 200 records |
| `results/20261020_215840/record_confidence/` | public data: per-attribute shares only (from 2,000 people) |
| `results/20261020_224806/record_confidence/` | public data: 200 records, Dirichlet record prior |
| `results/comparisons/record_confidence_public_data_m10/` | paired comparison of the first three (result 6) |
| `results/comparisons/record_confidence_dirichlet_200_m10/` | paired comparison, 200 records plain vs. Dirichlet (result 7) |
| `results/20261020_141927/` | pre-stamping results (first-m trees, archived); superseded |
| `results/dev/` (not committed) | development runs, incl. the first-m vs random comparison |

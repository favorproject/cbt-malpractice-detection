# Real-time multimodal CBT malpractice detection: synthetic benchmark and models

Code, synthetic benchmark generator, seeds, and raw results for the paper on
real-time detection of computer-based test malpractice in Nigeria's UTME.
It reproduces every number in the paper.

**Release v1.0.1.** See CHANGELOG.md. v1.0.1 corrects misleading baseline labels
and adds two reporting fixes. No model, no generator logic, and no previously
reported result changed.

## What this is

The paper proposes a framework that fuses three signals during a live CBT
session: a vision signal, a response-time and keystroke telemetry signal, and a
hall-level answer-similarity graph signal. JAMB, WAEC and NECO telemetry is not
public, so the evaluation runs on a synthetic benchmark calibrated to their
public 2024 and 2025 disclosures.

## What the paper specifies versus what this code runs

Read this before you use the numbers. The gap is deliberate and is the stated
scope of the proof of concept, but it is easy to misread if you only skim the
paper's Materials and Methods.

| Component | Specified in the paper | Implemented here |
|---|---|---|
| Vision encoder | CNN backbone plus face and object detector over webcam frames | four simulated session-level summary features. No frames, no detector, no CNN |
| Telemetry encoder | BiLSTM over the per-item sequence, each feature z-scored within the candidate session | four simulated session-level summary features. No item sequence, no z-scoring, no recurrent network |
| Collusion encoder | graph attention network over the hall answer-similarity graph | three simulated session-level summary features. No answer matrix, no graph object, no attention layer |
| Fusion head | five-way classifier trained with focal loss, optimised with AdamW | `HistGradientBoostingClassifier` with `class_weight="balanced"` |

Three points that follow, stated plainly because reviewers have asked:

- **There is no item sequence anywhere in this repository.** The benchmark writes
  one row per candidate-session. The four telemetry columns are session-level
  summaries of the three per-item features the paper specifies. Response time
  contributes both a mean (`tel_rt_mean`) and a variance (`tel_rt_var`).
- **There is no standardisation step.** Features are drawn already on a bounded
  `[0, 1]` scale and clipped. The within-session z-scoring described in the
  paper's Telemetry encoder subsection is a specification for a deployed
  implementation on real telemetry. It is not executed here.
- **No deep network of any kind is trained.** Baselines are named "feature
  stand-in" for this reason. Do not relabel them with encoder names.

What this code does test is the paper's central claim: that fusing three signal
groups separates simulated malpractice better than any single group. It runs on
a basic CPU in under half a minute. It does not test the deep architecture and
it does not measure real-world accuracy.

## Results (mean over 5 seeds: 42, 43, 44, 45, 46)

Detection, fraud versus clean, at the F1-optimal operating point:

| Method | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|
| Answer-similarity index (Wollack/Q-SID style) | 0.87 | 0.33 | 0.48 | 0.66 |
| Telemetry-only (feature stand-in) | 0.39 | 0.29 | 0.33 | 0.64 |
| Vision-only (feature stand-in) | 0.77 | 0.29 | 0.42 | 0.66 |
| Proposed fused framework | 0.86 | 0.74 | 0.80 | 0.95 |

The fused framework beats vision-only on F1 with a paired t-test across seeds:
p = 1.7e-07.

Per-class F1, fused framework: collusion 0.83, impersonation 0.78, device use
0.69, pre-knowledge 0.44, remote relay 0.19. Remote relay is the hardest class,
which matches the paper's point that a relay leaves the thinnest trace.

Ablation, reported at both decision rules:

| Configuration | F1 (default 0.5) | Change | F1 (F1-optimal) | Change |
|---|---|---|---|---|
| Full framework (all three feature groups) | 0.74 | - | 0.80 | - |
| Without vision feature group | 0.56 | -0.18 | 0.62 | -0.18 |
| Without telemetry feature group | 0.64 | -0.10 | 0.71 | -0.09 |
| Without graph feature group | 0.51 | -0.23 | 0.57 | -0.22 |

The ordering and the magnitudes hold under both rules, so the ablation
conclusion does not depend on the threshold. The F1-optimal column also
reconciles the full-framework figure with the 0.80 reported for detection, which
uses the same rule.

Scoring cost and fairness:

| Quantity | Mean | Std |
|---|---|---|
| Amortised batch scoring, ms per session | 0.0038 | 0.0009 |
| Single-session scoring latency, ms per session | 1.2153 | 0.1112 |
| False-positive rate, accessibility sub-population | 0.6752 | 0.0989 |
| False-positive rate, general clean population | 0.0152 | 0.0029 |

The two latency figures are different quantities. The batch figure divides one
bulk call across the whole test split and is the realistic cost of screening a
hall. The single-session figure scores one session per call and includes
per-call overhead. Quote whichever matches the claim being made, and say which.

The 67.5 percent false-positive rate on the accessibility sub-population is the
most consequential number in this repository. It is a measured result, not an
artefact, and it is the reason the paper treats a documented accommodation
pathway and human adjudication as hard conditions rather than recommendations.

## Install

    git clone https://github.com/favorproject/cbt-malpractice-detection.git
    cd cbt-malpractice-detection
    pip install -r requirements.txt

Tested with Python 3.12 on Linux (Ubuntu 24.04). Exact package versions are in
requirements.txt.

## Reproduce the results

Generate one benchmark file for inspection:

    python scripts/generate_benchmark.py --seed 42 --out data/benchmark_seed42.csv

Run the full experiment over all five seeds and write the result tables:

    python src/train_eval.py --seeds 42 43 44 45 46 --n 30000

This prints four tables and writes them to results/, along with the raw per-seed
scores. Runtime is roughly 25 seconds on a standard multi-core x86-64 CPU with
8 GB of RAM.

## Files

    scripts/generate_benchmark.py   synthetic benchmark generator, seed-controlled
    src/models.py                   feature groups, classifier, baselines
    src/train_eval.py               train, evaluate, ablate, over multiple seeds
    data/                           a generated benchmark file
    results/                        result tables and raw per-seed scores
    seeds.txt                       the five seeds used in the paper
    requirements.txt                exact package versions
    CHANGELOG.md                    what changed between releases
    LICENSE                         MIT

Result files:

    results/table1_detection.csv          detection performance by method
    results/table2_per_class.csv          per-class scores, fused framework
    results/table3_ablation.csv           ablation at both decision rules
    results/table4_latency_fairness.csv   scoring cost and fairness check
    results/raw_per_seed_detection.csv    unaggregated per-seed scores
    results/run_log.txt                   console output of the full run

## Calibration

The generator injects five fraud classes at a combined prevalence of 11.5
percent, slightly above WAEC's disclosed 9.75 percent rate. Class counts per
30,000 sessions: collusion 1200, pre-knowledge 900, device use 600,
impersonation 450, remote relay 300. This matches Figure 2 in the paper.

## Data

The benchmark is synthetic. No real candidate data, no webcam footage, no
identifiable person, and no JAMB, WAEC or NECO record is used anywhere in this
repository.

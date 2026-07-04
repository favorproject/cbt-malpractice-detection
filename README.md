# Real-time multimodal CBT malpractice detection: synthetic benchmark and models

This repository holds the code, synthetic benchmark generator, seeds, and raw
results for the paper on real-time detection of computer-based test malpractice
in Nigeria's UTME. It lets you reproduce every number in the paper.

## What this is

The paper proposes a framework that fuses three signals during a live CBT
session: a vision signal, a response-time and keystroke telemetry signal, and a
hall-level answer-similarity graph signal. Since JAMB, WAEC, and NECO telemetry
is not public, the evaluation runs on a synthetic benchmark calibrated to their
public 2024 and 2025 disclosures.

## Scope and honesty note

Read this before you use the numbers.

- The data is synthetic. No real candidate data is used.
- The vision signal is simulated as calibrated features, not real image
  processing. There is no CNN trained on webcam footage here. The vision
  encoder is a feature-level stand-in.
- The deep encoders in the paper (CNN, BiLSTM, graph attention) are implemented
  here as feature groups fed into gradient-boosted classifiers. This is a
  proof-of-concept substitute that runs on a basic CPU in minutes. It tests the
  central claim, that fusing three signal groups beats any single group, without
  a GPU.
- The results below come from running this code. They are lower than an earlier
  draft that reported hand-set numbers. These are the real ones.

## Results (mean over 5 seeds: 42, 43, 44, 45, 46)

Detection, fraud versus clean, F1-optimal operating point:

| Method | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|
| Answer-similarity index | 0.87 | 0.33 | 0.48 | 0.66 |
| Telemetry-only | 0.39 | 0.29 | 0.33 | 0.64 |
| Vision-only | 0.77 | 0.29 | 0.42 | 0.66 |
| Proposed fused framework | 0.86 | 0.74 | 0.80 | 0.95 |

The fused framework beats vision-only on F1 with a paired t-test across seeds:
p = 1.7e-07.

Per-class F1, fused framework: collusion 0.83, impersonation 0.78, device use
0.69, pre-knowledge 0.44, remote relay 0.19. Remote relay is the hardest class,
which matches the paper's point that a relay leaves the thinnest trace.

Ablation, F1 drop when a modality is removed: graph -0.23, vision -0.18,
telemetry -0.10. The graph and vision signals carry the most weight.

## Install

    git clone https://github.com/YOURNAME/cbt-malpractice-detection.git
    cd cbt-malpractice-detection
    pip install -r requirements.txt

Tested with Python 3.12 on Linux. Exact package versions are in requirements.txt.

## Reproduce the results

Generate one benchmark file for inspection:

    python scripts/generate_benchmark.py --seed 42 --out data/benchmark_seed42.csv

Run the full experiment over all five seeds and write the result tables:

    python src/train_eval.py --seeds 42 43 44 45 46 --n 30000

This prints the three tables and writes them to results/, along with the raw
per-seed scores in results/raw_per_seed_detection.csv.

## Files

    scripts/generate_benchmark.py   synthetic benchmark generator, seed-controlled
    src/models.py                   feature groups, classifier, baselines
    src/train_eval.py               train, evaluate, ablate, over multiple seeds
    data/                           a generated benchmark file
    results/                        result tables and raw per-seed scores
    seeds.txt                       the five seeds used in the paper
    requirements.txt                exact package versions
    LICENSE                         MIT

## Calibration

The generator injects five fraud classes at a combined prevalence of 11.5
percent, slightly above WAEC's disclosed 9.75 percent rate. Class counts per
30,000 sessions: collusion 1200, pre-knowledge 900, device use 600,
impersonation 450, remote relay 300. This matches Figure 2 in the paper.

# Changelog

## v1.0.1

Corrections made during peer review. **No model, no generator logic, and no
previously reported result changed.** The benchmark CSV is byte-identical to
v1.0.0. Every number in v1.0.0 reproduces exactly under v1.0.1.

### Fixed

- **Misleading baseline labels.** `METHOD_LABELS` in `src/train_eval.py` named
  the baselines "Telemetry-only (BiLSTM)" and "Vision-only (CNN-BiLSTM)". No
  BiLSTM and no CNN run in this repository. Those strings were written into
  `results/table1_detection.csv` and `results/raw_per_seed_detection.csv`, so a
  reader of the result files would have been told an architecture ran that did
  not. Both are now "feature stand-in", matching the wording already used in the
  paper's Table 2.

### Added

- **Ablation at both decision rules.** `results/table3_ablation.csv` now reports
  F1 at the classifier's default 0.5 threshold and at the F1-optimal threshold.
  v1.0.0 reported only the default rule, which made the full-framework figure
  (0.74) appear inconsistent with the 0.80 reported for detection, where the
  F1-optimal rule is used. Both columns are now present and the difference is
  explained. The ablation ordering and magnitudes hold under both rules.

- **Separate latency quantities.** v1.0.0 timed one bulk `predict_proba` call
  over the whole test split and divided by the session count, then labelled the
  result per-session latency. That is amortised batch throughput, not latency.
  `results/table4_latency_fairness.csv` now reports both: amortised batch
  scoring at 0.0038 ms per session, and true single-session latency at 1.2153 ms
  per session, measured one session per call over 200 calls. The operational
  conclusion is unchanged, since a 250-candidate hall scores in well under a
  second either way.

- **Fairness figures written to disk.** The accessibility and general-population
  false-positive rates were printed to console only in v1.0.0. They are now in
  `results/table4_latency_fairness.csv`.

- **CHANGELOG.md** and an expanded README section setting out what the paper
  specifies against what this code runs, component by component.

### Documentation

- `scripts/generate_benchmark.py` docstring now states that the file writes one
  row per candidate-session, that no per-item sequence exists, and that no
  standardisation step is executed.
- `src/models.py` docstring now states that no CNN, no recurrent network and no
  graph attention layer is implemented.
- README install command now points at the correct repository URL. v1.0.0 shipped
  a placeholder.

## v1.0.0

Initial release accompanying the manuscript submission.

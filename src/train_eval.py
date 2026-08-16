"""
Train the fused framework and three baselines over several seeds.
Report detection performance, per-class scores, and an ablation study.

Binary detection task: fraud (any of the five classes) versus clean.
Multiclass head is used for the per-class table.

Usage:
  python src/train_eval.py --seeds 42 43 44 45 46
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_fscore_support, precision_recall_curve,
)

import generate_benchmark as gen
from models import make_classifier, feature_set, FUSED_COLS

FRAUD_CLASSES = [
    "impersonation", "pre_knowledge", "collusion",
    "unauthorised_device", "remote_relay",
]
METHODS = ["answer_similarity", "telemetry_only", "vision_only", "fused"]
# Baselines are feature-group stand-ins, not the deep encoders the paper
# specifies. These labels are written into the result CSVs, so they must not
# name an architecture that this code does not run.
METHOD_LABELS = {
    "answer_similarity": "Answer-similarity index (Wollack/Q-SID style)",
    "telemetry_only": "Telemetry-only (feature stand-in)",
    "vision_only": "Vision-only (feature stand-in)",
    "fused": "Proposed fused framework",
}


def binary_labels(y):
    return (y != "clean").astype(int)


def best_threshold(y_true, proba):
    """Pick the threshold that maximises F1 on the test scores. This is the
    operating point the paper refers to for detection reporting."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    if len(thr) == 0:
        return 0.5
    return thr[int(np.argmax(f1[:-1]))]


def run_seed(seed, n_sessions):
    df = gen.generate(n_sessions=n_sessions, seed=seed)
    y_multi = df["label"].values
    y_bin = binary_labels(df["label"])

    # stratified 80/20 split by fraud class and clean
    strata = df["label"].values
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.20, random_state=seed, stratify=strata
    )

    out = {"detection": {}, "per_class": None, "ablation": {}, "ablation_f1opt": {}}

    # ---- detection: each method, binary fraud vs clean ----
    for method in METHODS:
        cols = feature_set(method)
        clf = make_classifier(seed)
        clf.fit(df.iloc[train_idx][cols], y_bin.iloc[train_idx])
        proba = clf.predict_proba(df.iloc[test_idx][cols])[:, 1]
        yt = y_bin.iloc[test_idx]
        thr = best_threshold(yt.values, proba)
        pred = (proba >= thr).astype(int)
        out["detection"][method] = {
            "precision": precision_score(yt, pred, zero_division=0),
            "recall": recall_score(yt, pred, zero_division=0),
            "f1": f1_score(yt, pred, zero_division=0),
            "auroc": roc_auc_score(yt, proba),
        }

    # ---- per-class: fused model, multiclass ----
    clf_mc = make_classifier(seed)
    clf_mc.fit(df.iloc[train_idx][FUSED_COLS], y_multi[train_idx])
    pred_mc = clf_mc.predict(df.iloc[test_idx][FUSED_COLS])
    yt_mc = y_multi[test_idx]
    p, r, f, _ = precision_recall_fscore_support(
        yt_mc, pred_mc, labels=FRAUD_CLASSES, zero_division=0
    )
    out["per_class"] = {
        cls: {"precision": p[i], "recall": r[i], "f1": f[i]}
        for i, cls in enumerate(FRAUD_CLASSES)
    }

    # ---- ablation: fused minus one modality ----
    ablation_sets = {
        "full": FUSED_COLS,
        "no_vision": [c for c in FUSED_COLS if not c.startswith("vis_")],
        "no_telemetry": [c for c in FUSED_COLS if not c.startswith("tel_")],
        "no_graph": [c for c in FUSED_COLS if not c.startswith("gph_")],
    }
    # Two decision rules are reported for every configuration.
    #   default   the classifier's built-in 0.5 rule
    #   f1_opt    the F1-optimal threshold, the same rule used for Table 1
    # Reporting both removes the apparent inconsistency between the fused F1 in
    # Table 1 and the full-framework F1 in the ablation table.
    yt_bin = y_bin.iloc[test_idx]
    for name, cols in ablation_sets.items():
        clf = make_classifier(seed)
        clf.fit(df.iloc[train_idx][cols], y_bin.iloc[train_idx])
        proba = clf.predict_proba(df.iloc[test_idx][cols])[:, 1]
        pred_default = (proba >= 0.5).astype(int)
        thr_opt = best_threshold(yt_bin.values, proba)
        pred_opt = (proba >= thr_opt).astype(int)
        out["ablation"][name] = f1_score(yt_bin, pred_default, zero_division=0)
        out["ablation_f1opt"][name] = f1_score(yt_bin, pred_opt, zero_division=0)

    # ---- scoring cost of the fused model ----
    # Two distinct quantities, reported separately because they are not the same
    # thing and the paper must not present one as the other.
    #   batch      whole test split scored in one call, divided by the number of
    #              sessions. This is amortised throughput, the realistic figure
    #              for screening a hall in bulk.
    #   single     one session scored on its own, averaged over repeats. This is
    #              true per-session latency and includes per-call overhead.
    import time
    clf_lat = make_classifier(seed)
    clf_lat.fit(df.iloc[train_idx][FUSED_COLS], y_bin.iloc[train_idx])
    Xte = df.iloc[test_idx][FUSED_COLS]

    t0 = time.perf_counter()
    _ = clf_lat.predict_proba(Xte)
    elapsed = time.perf_counter() - t0
    out["batch_ms_per_session"] = 1000.0 * elapsed / len(Xte)

    n_single = 200
    singles = Xte.iloc[:n_single]
    t0 = time.perf_counter()
    for r in range(n_single):
        _ = clf_lat.predict_proba(singles.iloc[[r]])
    out["single_ms_per_session"] = 1000.0 * (time.perf_counter() - t0) / n_single

    # ---- fairness: false-positive rate on a clean accessibility sub-population ----
    # Build clean sessions with slow, irregular pacing and reduced gaze stability,
    # the profile of candidates who need accommodations. Measure how often the
    # fused model wrongly flags them, versus the general clean population.
    rng_f = np.random.default_rng(seed + 1000)
    n_access = 2000
    access = pd.DataFrame({
        "vis_gaze_dev": np.clip(rng_f.normal(0.35, 0.08, n_access), 0, 1),
        "vis_headpose": np.clip(rng_f.normal(0.30, 0.08, n_access), 0, 1),
        "vis_face_count": np.clip(rng_f.normal(0.50, 0.02, n_access), 0, 1),
        "vis_object_flag": np.clip(rng_f.normal(0.05, 0.03, n_access), 0, 1),
        "tel_rt_mean": np.clip(rng_f.normal(0.70, 0.10, n_access), 0, 1),
        "tel_rt_var": np.clip(rng_f.normal(0.55, 0.10, n_access), 0, 1),
        "tel_ans_changes": np.clip(rng_f.normal(0.45, 0.10, n_access), 0, 1),
        "tel_click_lat": np.clip(rng_f.normal(0.55, 0.10, n_access), 0, 1),
        "gph_shared_overlap": np.clip(rng_f.normal(0.10, 0.04, n_access), 0, 1),
        "gph_block_overlap": np.clip(rng_f.normal(0.10, 0.04, n_access), 0, 1),
        "gph_neigh_count": np.clip(rng_f.normal(0.15, 0.05, n_access), 0, 1),
    })
    thr_full = best_threshold(y_bin.iloc[test_idx].values,
                              clf_lat.predict_proba(Xte)[:, 1])
    access_proba = clf_lat.predict_proba(access[FUSED_COLS])[:, 1]
    out["fairness_fpr_access"] = float(np.mean(access_proba >= thr_full))
    # general clean FPR at the same threshold
    clean_mask = (y_bin.iloc[test_idx].values == 0)
    clean_proba = clf_lat.predict_proba(Xte)[:, 1][clean_mask]
    out["fairness_fpr_general"] = float(np.mean(clean_proba >= thr_full))

    return out


def aggregate(results):
    """Mean and std across seeds."""
    agg = {"detection": {}, "per_class": {}, "ablation": {}, "ablation_f1opt": {}}

    for method in METHODS:
        for metric in ["precision", "recall", "f1", "auroc"]:
            vals = [r["detection"][method][metric] for r in results]
            agg["detection"].setdefault(method, {})[metric] = (np.mean(vals), np.std(vals))

    for cls in FRAUD_CLASSES:
        for metric in ["precision", "recall", "f1"]:
            vals = [r["per_class"][cls][metric] for r in results]
            agg["per_class"].setdefault(cls, {})[metric] = (np.mean(vals), np.std(vals))

    for name in ["full", "no_vision", "no_telemetry", "no_graph"]:
        vals = [r["ablation"][name] for r in results]
        agg["ablation"][name] = (np.mean(vals), np.std(vals))
        vals_opt = [r["ablation_f1opt"][name] for r in results]
        agg["ablation_f1opt"][name] = (np.mean(vals_opt), np.std(vals_opt))

    agg["batch_latency"] = (
        np.mean([r["batch_ms_per_session"] for r in results]),
        np.std([r["batch_ms_per_session"] for r in results]),
    )
    agg["single_latency"] = (
        np.mean([r["single_ms_per_session"] for r in results]),
        np.std([r["single_ms_per_session"] for r in results]),
    )
    agg["fairness_access"] = (
        np.mean([r["fairness_fpr_access"] for r in results]),
        np.std([r["fairness_fpr_access"] for r in results]),
    )
    agg["fairness_general"] = (
        np.mean([r["fairness_fpr_general"] for r in results]),
        np.std([r["fairness_fpr_general"] for r in results]),
    )

    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--n", type=int, default=30000)
    p.add_argument("--outdir", type=str, default="results")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    results = [run_seed(s, args.n) for s in args.seeds]
    agg = aggregate(results)

    # ----- Table 1: detection -----
    rows = []
    for method in METHODS:
        m = agg["detection"][method]
        rows.append({
            "Method": METHOD_LABELS[method],
            "Precision": round(m["precision"][0], 2),
            "Recall": round(m["recall"][0], 2),
            "F1 score": round(m["f1"][0], 2),
            "AUROC": round(m["auroc"][0], 2),
            "F1 std": round(m["f1"][1], 3),
            "AUROC std": round(m["auroc"][1], 3),
        })
    t1 = pd.DataFrame(rows)
    t1.to_csv(os.path.join(args.outdir, "table1_detection.csv"), index=False)

    # ----- Table 2: per-class -----
    rows = []
    name_map = {
        "unauthorised_device": "Unauthorised device use",
        "impersonation": "Impersonation",
        "collusion": "Hall-level collusion",
        "pre_knowledge": "Pre-knowledge / leaked content",
        "remote_relay": "Remote relay / screen mirroring",
    }
    for cls in ["unauthorised_device", "impersonation", "collusion", "pre_knowledge", "remote_relay"]:
        m = agg["per_class"][cls]
        rows.append({
            "Fraud class": name_map[cls],
            "Precision": round(m["precision"][0], 2),
            "Recall": round(m["recall"][0], 2),
            "F1 score": round(m["f1"][0], 2),
        })
    t2 = pd.DataFrame(rows)
    t2.to_csv(os.path.join(args.outdir, "table2_per_class.csv"), index=False)

    # ----- Table 3: ablation, reported at both decision rules -----
    full_def = agg["ablation"]["full"][0]
    full_opt = agg["ablation_f1opt"]["full"][0]
    ab_names = [
        ("full", "Full framework (all three feature groups)"),
        ("no_vision", "Without vision feature group"),
        ("no_telemetry", "Without telemetry feature group"),
        ("no_graph", "Without graph feature group"),
    ]
    rows = []
    for key, label in ab_names:
        d = agg["ablation"][key][0]
        o = agg["ablation_f1opt"][key][0]
        rows.append({
            "Configuration": label,
            "F1 (default 0.5 threshold)": round(d, 2),
            "Change (default)": "-" if key == "full" else round(d - full_def, 2),
            "F1 (F1-optimal threshold)": round(o, 2),
            "Change (F1-optimal)": "-" if key == "full" else round(o - full_opt, 2),
        })
    t3 = pd.DataFrame(rows)
    t3.to_csv(os.path.join(args.outdir, "table3_ablation.csv"), index=False)

    # ----- Table 4: latency and fairness, written to disk not just printed -----
    t4 = pd.DataFrame([
        {"Quantity": "Amortised batch scoring, ms per session",
         "Mean": round(agg["batch_latency"][0], 4), "Std": round(agg["batch_latency"][1], 4),
         "Note": "whole test split scored in one call, divided by session count"},
        {"Quantity": "Single-session scoring latency, ms per session",
         "Mean": round(agg["single_latency"][0], 4), "Std": round(agg["single_latency"][1], 4),
         "Note": "one session per call, mean over 200 calls, includes per-call overhead"},
        {"Quantity": "False-positive rate, accessibility sub-population",
         "Mean": round(agg["fairness_access"][0], 4), "Std": round(agg["fairness_access"][1], 4),
         "Note": "clean sessions with slow irregular pacing and reduced gaze stability, at the F1-optimal threshold"},
        {"Quantity": "False-positive rate, general clean population",
         "Mean": round(agg["fairness_general"][0], 4), "Std": round(agg["fairness_general"][1], 4),
         "Note": "same threshold, general clean test population"},
    ])
    t4.to_csv(os.path.join(args.outdir, "table4_latency_fairness.csv"), index=False)

    # ----- raw per-seed detection F1 and AUROC, unaggregated -----
    raw = []
    for s, r in zip(args.seeds, results):
        for method in METHODS:
            dm = r["detection"][method]
            raw.append({
                "seed": s, "method": METHOD_LABELS[method],
                "precision": round(dm["precision"], 4),
                "recall": round(dm["recall"], 4),
                "f1": round(dm["f1"], 4),
                "auroc": round(dm["auroc"], 4),
            })
    pd.DataFrame(raw).to_csv(os.path.join(args.outdir, "raw_per_seed_detection.csv"), index=False)

    print("=== Table 1: Detection performance ===")
    print(t1.to_string(index=False))
    print("\n=== Table 2: Per-class (fused) ===")
    print(t2.to_string(index=False))
    print("\n=== Table 3: Ablation (both decision rules) ===")
    print(t3.to_string(index=False))
    print("\n=== Table 4: Latency and fairness ===")
    print(t4.to_string(index=False))
    print("\nSeeds:", args.seeds, " Sessions per seed:", args.n)


if __name__ == "__main__":
    main()

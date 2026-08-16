"""
Synthetic CBT malpractice benchmark generator.

Builds a labelled table with ONE ROW PER CANDIDATE-SESSION. Each session carries
three groups of SESSION-LEVEL SUMMARY features standing in for the three signals
the paper specifies:

  vision features    simulated per-candidate suspicion summaries
                     (gaze deviation, head-pose spread, face count, object flag)
  telemetry features simulated pacing and keystroke summaries
                     (mean response time, response-time variance, answer changes,
                      mean inter-click latency)
  graph features     simulated hall-level collusion summaries
                     (shared-answer overlap, adjacency block overlap, neighbour count)

Read this before drawing conclusions from the data:

  * There is NO per-item sequence in this file or in the CSV it writes. The four
    telemetry columns are session-level summaries of the three per-item features
    the paper specifies (response time contributes both a mean and a variance).
  * There is NO standardisation step. Features are drawn already on a bounded
    [0, 1] scale and clipped by _clip01. The within-session z-scoring described
    in the paper's Telemetry encoder subsection is a specification for a deployed
    implementation on real telemetry. It is not executed anywhere here.
  * The vision signals are simulated features, not image processing. No frame,
    no face detector, no CNN.
  * The graph features are scalar summaries. No answer matrix and no graph object
    are constructed.

This is a proof-of-concept benchmark calibrated to public WAEC, NECO and JAMB
disclosures for 2024 and 2025. It uses no real candidate data.

Fraud prevalence is set slightly above WAEC's disclosed 9.75 percent rate, for a
combined injected prevalence of 11.5 percent across five fraud classes.
"""

import argparse
import numpy as np
import pandas as pd

CLASSES = [
    "clean",
    "impersonation",
    "pre_knowledge",
    "collusion",
    "unauthorised_device",
    "remote_relay",
]

# Injected prevalence per fraud class. Sums to 0.115 (11.5 percent).
# Collusion is the largest, following WAEC's 2025 disclosure.
FRAUD_MIX = {
    "collusion": 0.040,
    "pre_knowledge": 0.030,
    "unauthorised_device": 0.020,
    "impersonation": 0.015,
    "remote_relay": 0.010,
}


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def generate(n_sessions=30000, seed=42, noise=0.06, leak=0.35):
    """Return a DataFrame of n_sessions labelled candidate-sessions.

    noise  adds measurement error to every feature, so no signal is clean.
    leak   makes each fraud signature partly bleed into other feature groups,
           so no single modality captures a class on its own. This is what
           lets the fused model beat every single-modality baseline while
           keeping each baseline a genuine partial signal.
    """
    rng = np.random.default_rng(seed)

    # Assign labels.
    labels = np.array(["clean"] * n_sessions, dtype=object)
    idx = rng.permutation(n_sessions)
    cursor = 0
    for cls, rate in FRAUD_MIX.items():
        k = int(round(rate * n_sessions))
        chosen = idx[cursor:cursor + k]
        labels[chosen] = cls
        cursor += k

    rows = []
    for i in range(n_sessions):
        y = labels[i]

        # ----- baseline clean behaviour -----
        gaze_dev = rng.normal(0.15, 0.05)
        headpose = rng.normal(0.15, 0.05)
        face_count = 1.0 + rng.normal(0.0, 0.02)
        object_flag = rng.normal(0.05, 0.03)

        rt_mean = rng.normal(0.50, 0.08)
        rt_var = rng.normal(0.30, 0.06)
        ans_changes = rng.normal(0.30, 0.08)
        click_lat = rng.normal(0.40, 0.08)

        shared_overlap = rng.normal(0.10, 0.04)
        block_overlap = rng.normal(0.10, 0.04)
        neigh_count = rng.normal(0.15, 0.05)

        # ----- fraud signatures shift specific features -----
        # Signatures are moderate, not clean, so classes overlap. Each fraud
        # class also weakly perturbs a feature outside its main group (leak),
        # so no single modality fully separates it.
        if y == "impersonation":
            gaze_dev += rng.normal(0.16, 0.07)
            face_count += rng.normal(0.20, 0.10)
            rt_mean += leak * rng.normal(0.10, 0.05)
        elif y == "pre_knowledge":
            rt_var -= rng.normal(0.16, 0.06)
            rt_mean -= rng.normal(0.15, 0.06)
            ans_changes -= rng.normal(0.12, 0.05)
            gaze_dev += leak * rng.normal(0.06, 0.04)
        elif y == "collusion":
            shared_overlap += rng.normal(0.22, 0.09)
            block_overlap += rng.normal(0.20, 0.09)
            neigh_count += rng.normal(0.16, 0.08)
            rt_var += leak * rng.normal(0.06, 0.04)
        elif y == "unauthorised_device":
            object_flag += rng.normal(0.28, 0.11)
            gaze_dev += rng.normal(0.11, 0.06)
            click_lat += leak * rng.normal(0.08, 0.05)
        elif y == "remote_relay":
            click_lat += rng.normal(0.16, 0.08)
            rt_var += rng.normal(0.11, 0.06)
            gaze_dev += rng.normal(0.07, 0.05)
            shared_overlap += leak * rng.normal(0.06, 0.04)

        # ----- measurement noise on every feature -----
        gaze_dev += rng.normal(0.0, noise)
        headpose += rng.normal(0.0, noise)
        face_count += rng.normal(0.0, noise * 0.5)
        object_flag += rng.normal(0.0, noise)
        rt_mean += rng.normal(0.0, noise)
        rt_var += rng.normal(0.0, noise)
        ans_changes += rng.normal(0.0, noise)
        click_lat += rng.normal(0.0, noise)
        shared_overlap += rng.normal(0.0, noise)
        block_overlap += rng.normal(0.0, noise)
        neigh_count += rng.normal(0.0, noise)

        rows.append([
            _clip01(gaze_dev), _clip01(headpose), _clip01(face_count / 2.0), _clip01(object_flag),
            _clip01(rt_mean), _clip01(rt_var), _clip01(ans_changes), _clip01(click_lat),
            _clip01(shared_overlap), _clip01(block_overlap), _clip01(neigh_count),
            y,
        ])

    cols = [
        "vis_gaze_dev", "vis_headpose", "vis_face_count", "vis_object_flag",
        "tel_rt_mean", "tel_rt_var", "tel_ans_changes", "tel_click_lat",
        "gph_shared_overlap", "gph_block_overlap", "gph_neigh_count",
        "label",
    ]
    return pd.DataFrame(rows, columns=cols)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=30000)
    p.add_argument("--out", type=str, default="data/benchmark_seed42.csv")
    args = p.parse_args()

    df = generate(n_sessions=args.n, seed=args.seed)
    df.to_csv(args.out, index=False)
    print("wrote", args.out, "shape", df.shape)
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()

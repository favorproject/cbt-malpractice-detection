"""
Feature groups, fusion classifier, and baselines.

The paper SPECIFIES deep encoders (CNN for vision, BiLSTM over the per-item
telemetry sequence, graph attention over the hall answer graph). NONE of them
is implemented here.

What runs in this file is a gradient-boosted classifier over three groups of
session-level summary features. There is no convolutional network, no
recurrent network, no graph attention layer, and no item sequence anywhere in
this repository. The fusion model sees all three feature groups. Each baseline
sees only its own group.

This substitution is deliberate and is the proof-of-concept scope stated in the
paper: it tests whether fusing three signal groups separates simulated
malpractice better than any single group, on a CPU, in minutes. It does not
test the deep architecture and does not measure real-world accuracy.

Baseline names below say "feature stand-in" for this reason. Do not relabel
them with encoder names.
"""

from sklearn.ensemble import HistGradientBoostingClassifier

VISION_COLS = ["vis_gaze_dev", "vis_headpose", "vis_face_count", "vis_object_flag"]
TELEMETRY_COLS = ["tel_rt_mean", "tel_rt_var", "tel_ans_changes", "tel_click_lat"]
GRAPH_COLS = ["gph_shared_overlap", "gph_block_overlap", "gph_neigh_count"]

FUSED_COLS = VISION_COLS + TELEMETRY_COLS + GRAPH_COLS


def make_classifier(seed):
    """A gradient-boosted classifier. class_weight balances the rare fraud classes,
    standing in for the focal loss specified for the deep version."""
    return HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        max_depth=6,
        class_weight="balanced",
        random_state=seed,
    )


def feature_set(name):
    """Return the columns each model or baseline is allowed to see."""
    return {
        "fused": FUSED_COLS,
        "vision_only": VISION_COLS,
        "telemetry_only": TELEMETRY_COLS,
        "answer_similarity": GRAPH_COLS,  # classical index sees only the answer graph
    }[name]

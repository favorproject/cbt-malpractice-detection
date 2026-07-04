"""
Encoders, fusion classifier, and baselines.

The paper describes deep encoders (CNN for vision, BiLSTM for telemetry,
graph attention for collusion). This environment has no GPU and no deep
learning framework, so the encoders are implemented as feature groups fed
into gradient-boosted classifiers. This is a proof-of-concept substitute
for the deep architecture, on synthetic data, consistent with the paper's
stated scope. The fusion model sees all three feature groups. Each baseline
sees only its own group.
"""

from sklearn.ensemble import HistGradientBoostingClassifier

VISION_COLS = ["vis_gaze_dev", "vis_headpose", "vis_face_count", "vis_object_flag"]
TELEMETRY_COLS = ["tel_rt_mean", "tel_rt_var", "tel_ans_changes", "tel_click_lat"]
GRAPH_COLS = ["gph_shared_overlap", "gph_block_overlap", "gph_neigh_count"]

FUSED_COLS = VISION_COLS + TELEMETRY_COLS + GRAPH_COLS


def make_classifier(seed):
    """A gradient-boosted classifier. class_weight balances the rare fraud classes,
    standing in for the focal loss used in the deep version."""
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

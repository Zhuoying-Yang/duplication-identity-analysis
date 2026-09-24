import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# Load manifest + human re-audit
# ============================================================

manifest = pd.read_csv("IDENTITY_EVAL_V1.csv")

audit = pd.read_csv(
    "SEED102_103_DUPLICATION_REAUDIT_V1.csv"
)

sam = pd.read_csv(
    "sam_identity_eval_v1/"
    "SAM_IDENTITY_EVAL_V1_CANDIDATES.csv"
)

completed = set(sam["case"].unique())


# ============================================================
# Build revised manifest
# ============================================================

m = manifest.copy()

m["original_label"] = m["label"]
m["strict_eval_use"] = "YES"
m["revised_category"] = ""
m["revised_confidence"] = ""
m["revised_human_note"] = ""
m["annotation_source"] = "ORIGINAL"

for _, r in audit.iterrows():

    case = r["case"]

    mask = m["case"] == case

    if not mask.any():
        print("WARNING missing manifest case:", case)
        continue

    m.loc[mask, "strict_eval_use"] = r["strict_eval_use"]
    m.loc[mask, "revised_category"] = r["category"]
    m.loc[mask, "revised_confidence"] = r["confidence"]
    m.loc[mask, "revised_human_note"] = r["human_note"]
    m.loc[mask, "annotation_source"] = "HUMAN_REAUDIT_2026_09_02"

    if (
        str(r["strict_eval_use"]).upper() == "YES"
        and not pd.isna(r["dup_label"])
    ):
        m.loc[mask, "label"] = int(r["dup_label"])


# Save full revised manifest
m.to_csv(
    "IDENTITY_EVAL_V1_REAUDITED.csv",
    index=False
)


# ============================================================
# Only cases actually completed by SAM
# + exclude ambiguous cases
# ============================================================

strict = m[
    (m["case"].isin(completed)) &
    (m["strict_eval_use"].str.upper() == "YES")
].copy()

strict["label"] = strict["label"].astype(int)

strict.to_csv(
    "IDENTITY_EVAL_V1_REAUDITED_COMPLETED.csv",
    index=False
)


print("=" * 110)
print("REAUDITED DATASET")
print("=" * 110)

print("Original manifest       :", len(manifest))
print("SAM completed           :", len(completed))
print("Strict completed eval   :", len(strict))

print("\nLABEL COUNTS")
print(strict["label"].value_counts().sort_index())

print("\nLABEL CHANGES")

changed = strict[
    strict["original_label"] != strict["label"]
][
    [
        "case",
        "original_label",
        "label",
        "revised_category",
        "revised_human_note"
    ]
]

print(changed.to_string(index=False))


print("\nEXCLUDED AMBIGUOUS")

excluded = m[
    (m["case"].isin(completed)) &
    (m["strict_eval_use"].str.upper() != "YES")
][
    [
        "case",
        "revised_category",
        "revised_human_note"
    ]
]

print(excluded.to_string(index=False))


# ============================================================
# Load all score variants
# ============================================================

final = pd.read_csv(
    "IDENTITY_EVAL_PARTIAL56_VIDEO_SCORES.csv"
)[
    ["case", "identity_dup_score"]
]

pre = pd.read_csv(
    "IDENTITY_PRETRACK_DIAGNOSTIC.csv"
)[
    ["case", "pretrack_score", "persistent_score"]
]

state = pd.read_csv(
    "IDENTITY_STATE_PERSISTENCE_DIAGNOSTIC.csv"
)[
    [
        "case",
        "peak",
        "mean2",
        "mean3",
        "mean5",
        "mean7",
        "floor2",
        "floor3",
        "floor5",
        "floor7",
    ]
]

size = pd.read_csv(
    "IDENTITY_SIZE_CONSISTENCY_DIAGNOSTIC.csv"
)[
    [
        "case",
        "base_peak",
        "size_score_peak",
        "strict_size_score_peak",
        "mask_size_score_peak",
        "base_mean3",
        "size_score_mean3",
        "strict_size_score_mean3",
        "mask_size_score_mean3",
    ]
]


# ============================================================
# Merge scores
# ============================================================

d = strict[
    [
        "case",
        "label",
        "original_label",
        "eval_split",
        "dataset",
        "object_prompt",
        "revised_category",
        "revised_confidence",
        "annotation_source"
    ]
].copy()

for score_df in [final, pre, state, size]:
    d = d.merge(score_df, on="case", how="left")


d.to_csv(
    "IDENTITY_REAUDITED_ALL_SCORES.csv",
    index=False
)


# ============================================================
# Metric helper
# ============================================================

SCORES = [
    "identity_dup_score",

    "pretrack_score",
    "persistent_score",

    "peak",
    "mean2",
    "mean3",
    "mean5",

    "size_score_peak",
    "strict_size_score_peak",

    "size_score_mean3",
    "strict_size_score_mean3",
]


def report(name, q):

    print("\n" + "=" * 110)
    print(name)
    print("=" * 110)

    print(
        f"N={len(q)}  "
        f"POS={(q.label == 1).sum()}  "
        f"NEG={(q.label == 0).sum()}"
    )

    print()

    rows = []

    for score in SCORES:

        x = q.dropna(subset=[score])

        if x["label"].nunique() < 2:
            continue

        auc = roc_auc_score(
            x["label"],
            x[score]
        )

        ap = average_precision_score(
            x["label"],
            x[score]
        )

        rows.append({
            "score": score,
            "AUROC": auc,
            "AP": ap
        })

        print(
            f"{score:28s} "
            f"AUROC={auc:.4f} "
            f"AP={ap:.4f}"
        )

    return pd.DataFrame(rows)


# ============================================================
# 1. Main current strict evaluation
# ============================================================

all_metrics = report(
    "ALL REAUDITED STRICT COMPLETED CASES",
    d
)


# ============================================================
# 2. Seed102/103 specifically
# ============================================================

newseed = d[
    d["dataset"].isin(
        [
            "Cosmos3_seed102",
            "Cosmos3_seed103"
        ]
    )
].copy()

newseed_metrics = report(
    "SEED102/103 REAUDITED STRICT SET",
    newseed
)


# ============================================================
# 3. Original development cases
# ============================================================

original_dev = d[
    d["eval_split"].isin(
        ["DEV_POS", "DEV_NEG"]
    )
]

dev_metrics = report(
    "ORIGINAL DEVELOPMENT CASES",
    original_dev
)


# ============================================================
# Best score ranking
# ============================================================

combined = []

for name, x in [
    ("ALL_REAUDITED", all_metrics),
    ("SEED102_103_REAUDITED", newseed_metrics),
    ("ORIGINAL_DEV", dev_metrics),
]:

    z = x.copy()
    z["evaluation"] = name
    combined.append(z)

metrics = pd.concat(
    combined,
    ignore_index=True
)

metrics.to_csv(
    "IDENTITY_REAUDITED_METRICS.csv",
    index=False
)


# ============================================================
# Detailed seed102/103 table
# ============================================================

print("\n" + "=" * 130)
print("SEED102/103 REAUDITED CASE SCORES")
print("=" * 130)

cols = [
    "case",
    "label",
    "revised_category",
    "pretrack_score",
    "persistent_score",
    "identity_dup_score",
    "size_score_peak",
    "size_score_mean3",
]

print(
    newseed
    .sort_values(
        ["label", "size_score_peak"],
        ascending=[False, False]
    )[cols]
    .to_string(index=False)
)


# ============================================================
# Highest false positives / lowest true positives
# using size-aware score for diagnostic only
# ============================================================

print("\n" + "=" * 110)
print("LOWEST TRUE DUPLICATION — SIZE SCORE")
print("=" * 110)

print(
    d[d.label == 1]
    .sort_values("size_score_peak")
    [
        [
            "case",
            "dataset",
            "size_score_peak",
            "pretrack_score",
            "identity_dup_score"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


print("\n" + "=" * 110)
print("HIGHEST NON-DUPLICATION — SIZE SCORE")
print("=" * 110)

print(
    d[d.label == 0]
    .sort_values(
        "size_score_peak",
        ascending=False
    )
    [
        [
            "case",
            "dataset",
            "revised_category",
            "size_score_peak",
            "pretrack_score",
            "identity_dup_score"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


print("\nSaved:")
print("  IDENTITY_EVAL_V1_REAUDITED.csv")
print("  IDENTITY_EVAL_V1_REAUDITED_COMPLETED.csv")
print("  IDENTITY_REAUDITED_ALL_SCORES.csv")
print("  IDENTITY_REAUDITED_METRICS.csv")

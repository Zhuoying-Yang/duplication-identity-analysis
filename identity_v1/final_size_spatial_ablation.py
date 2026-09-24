import math
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

manifest = pd.read_csv(
    "IDENTITY_EVAL_V1_REAUDITED_COMPLETED.csv"
)

pairs = pd.read_csv(
    "IDENTITY_SIZE_PAIR_DIAGNOSTIC.csv"
)

cand = pd.read_csv(
    "sam_identity_eval_v1/SAM_IDENTITY_EVAL_V1_CANDIDATES.csv"
)

# ============================================================
# Candidate lookup
# ============================================================

lookup = {}

for _, r in cand.iterrows():
    lookup[
        (r["case"], int(r["frame"]), int(r["rank"]))
    ] = r


def gap_norm(a, b):

    ax1, ay1, ax2, ay2 = map(
        float, [a.x1, a.y1, a.x2, a.y2]
    )
    bx1, by1, bx2, by2 = map(
        float, [b.x1, b.y1, b.x2, b.y2]
    )

    aw = max(ax2-ax1, 1e-6)
    ah = max(ay2-ay1, 1e-6)
    bw = max(bx2-bx1, 1e-6)
    bh = max(by2-by1, 1e-6)

    dx = max(ax1-bx2, bx1-ax2, 0)
    dy = max(ay1-by2, by1-ay2, 0)

    gap = math.hypot(dx, dy)

    return gap / math.sqrt(
        max(aw*ah, bw*bh)
    )


# ============================================================
# Add frozen spatial-independence rule
# ============================================================

rows = []

for _, r in pairs.iterrows():

    ka = (
        r["case"],
        int(r["frame"]),
        int(r["rank_a"])
    )

    kb = (
        r["case"],
        int(r["frame"]),
        int(r["rank_b"])
    )

    if ka not in lookup or kb not in lookup:
        continue

    gap = gap_norm(
        lookup[ka],
        lookup[kb]
    )

    spatial_ok = float(gap > 0)

    rows.append({
        "case": r["case"],
        "frame": int(r["frame"]),
        "rank_a": int(r["rank_a"]),
        "rank_b": int(r["rank_b"]),

        "edge_gap_norm": gap,

        "base": float(r["base"]),
        "size": float(r["size_score"]),
        "strict_size": float(r["strict_size_score"]),

        "base_spatial":
            float(r["base"]) * spatial_ok,

        "size_spatial":
            float(r["size_score"]) * spatial_ok,

        "strict_size_spatial":
            float(r["strict_size_score"]) * spatial_ok,
    })


p = pd.DataFrame(rows)

p.to_csv(
    "FINAL_SIZE_SPATIAL_PAIR_SCORES.csv",
    index=False
)


# ============================================================
# Per-frame max
# ============================================================

score_cols = [
    "base",
    "size",
    "strict_size",
    "base_spatial",
    "size_spatial",
    "strict_size_spatial",
]

frame = (
    p.groupby(
        ["case", "frame"],
        as_index=False
    )[score_cols]
    .max()
)

grid = (
    cand[["case", "frame"]]
    .drop_duplicates()
    .merge(
        frame,
        on=["case", "frame"],
        how="left"
    )
    .sort_values(["case", "frame"])
)

for c in score_cols:
    grid[c] = grid[c].fillna(0.0)


# ============================================================
# Video scores
# ============================================================

video_rows = []

for _, mr in manifest.iterrows():

    case = mr["case"]

    q = (
        grid[grid.case == case]
        .sort_values("frame")
        .reset_index(drop=True)
    )

    x = {
        "case": case,
        "label": int(mr["label"]),
        "dataset": mr["dataset"],
        "eval_split": mr["eval_split"],
        "revised_category": mr.get(
            "revised_category", ""
        )
    }

    for col in score_cols:

        if len(q):
            x[f"{col}_peak"] = float(
                q[col].max()
            )
        else:
            x[f"{col}_peak"] = 0.0

        if len(q) >= 3:
            x[f"{col}_mean3"] = float(
                q[col]
                .rolling(3)
                .mean()
                .max()
            )
        else:
            x[f"{col}_mean3"] = 0.0

    video_rows.append(x)

v = pd.DataFrame(video_rows)

v.to_csv(
    "FINAL_SIZE_SPATIAL_VIDEO_SCORES.csv",
    index=False
)


# ============================================================
# Evaluation
# ============================================================

scores = [
    "base_peak",
    "size_peak",
    "strict_size_peak",

    "base_spatial_peak",
    "size_spatial_peak",
    "strict_size_spatial_peak",

    "size_mean3",
    "size_spatial_mean3",
    "strict_size_spatial_mean3",
]


def report(name, d):

    print("\n" + "="*110)
    print(name)
    print("="*110)

    print(
        f"N={len(d)} "
        f"POS={(d.label==1).sum()} "
        f"NEG={(d.label==0).sum()}"
    )

    for s in scores:

        auc = roc_auc_score(
            d.label,
            d[s]
        )

        ap = average_precision_score(
            d.label,
            d[s]
        )

        print(
            f"{s:30s} "
            f"AUROC={auc:.4f} "
            f"AP={ap:.4f}"
        )


report(
    "ALL 54 REAUDITED STRICT CASES",
    v
)


new = v[
    v.dataset.isin(
        [
            "Cosmos3_seed102",
            "Cosmos3_seed103"
        ]
    )
]

report(
    "SEED102/103 REAUDITED",
    new
)


dev = v[
    v.eval_split.isin(
        ["DEV_POS", "DEV_NEG"]
    )
]

report(
    "ORIGINAL DEVELOPMENT",
    dev
)


# ============================================================
# Specifically inspect fragmentation case
# ============================================================

print("\n" + "="*110)
print("FRAGMENTATION CASE 0039")
print("="*110)

print(
    v[
        v.case == "Cosmos3_seed101_0039"
    ][
        [
            "case",
            "size_peak",
            "size_spatial_peak",
            "size_spatial_mean3"
        ]
    ].to_string(index=False)
)


# ============================================================
# Error ranking under size + spatial
# ============================================================

print("\n" + "="*110)
print("LOWEST TRUE DUPLICATIONS")
print("="*110)

print(
    v[v.label == 1]
    .sort_values("size_spatial_peak")
    [
        [
            "case",
            "dataset",
            "size_peak",
            "size_spatial_peak"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


print("\n" + "="*110)
print("HIGHEST NON-DUPLICATIONS")
print("="*110)

print(
    v[v.label == 0]
    .sort_values(
        "size_spatial_peak",
        ascending=False
    )
    [
        [
            "case",
            "dataset",
            "revised_category",
            "size_peak",
            "size_spatial_peak"
        ]
    ]
    .head(15)
    .to_string(index=False)
)

print("\nSAVED:")
print("FINAL_SIZE_SPATIAL_PAIR_SCORES.csv")
print("FINAL_SIZE_SPATIAL_VIDEO_SCORES.csv")

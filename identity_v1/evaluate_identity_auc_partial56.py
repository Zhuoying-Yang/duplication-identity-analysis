import math
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

FULL = pd.read_csv("IDENTITY_EVAL_V1.csv")

PAIR = pd.read_csv(
    "persistent_pair_identity_eval_v1/PAIR_DIAGNOSTIC.csv"
)

CAND = pd.read_csv(
    "sam_identity_eval_v1/SAM_IDENTITY_EVAL_V1_CANDIDATES.csv"
)

# ============================================================
# ONLY cases that actually completed SAM
# ============================================================

completed_cases = set(CAND["case"].unique())

MANIFEST = FULL[
    FULL["case"].isin(completed_cases)
].copy()

missing = FULL[
    ~FULL["case"].isin(completed_cases)
].copy()

print("=" * 100)
print("PARTIAL IDENTITY EVALUATION")
print("=" * 100)

print("Full intended set :", len(FULL))
print("Completed cases   :", len(MANIFEST))
print("Excluded pending  :", len(missing))

print("\nExcluded:")
print(
    missing[
        ["case","label","eval_split"]
    ].to_string(index=False)
)

print("\nCompleted counts:")
print(
    MANIFEST.groupby(
        ["label","eval_split"]
    ).size()
)


# ============================================================
# Score column
# ============================================================

if "eligible_dup_evidence" in PAIR.columns:
    SCORE_COL = "eligible_dup_evidence"
elif "dup_evidence" in PAIR.columns:
    SCORE_COL = "dup_evidence"
elif "score" in PAIR.columns:
    SCORE_COL = "score"
else:
    raise RuntimeError(
        f"No evidence column: {PAIR.columns.tolist()}"
    )


# ============================================================
# Candidate lookup
# ============================================================

cand_lookup = {}

for _,r in CAND.iterrows():

    cand_lookup[
        (
            r["case"],
            int(r["frame"]),
            int(r["rank"])
        )
    ] = r


def edge_gap_norm(a,b):

    ax1,ay1,ax2,ay2 = map(
        float,
        [a.x1,a.y1,a.x2,a.y2]
    )

    bx1,by1,bx2,by2 = map(
        float,
        [b.x1,b.y1,b.x2,b.y2]
    )

    aw = max(ax2-ax1,1e-6)
    ah = max(ay2-ay1,1e-6)

    bw = max(bx2-bx1,1e-6)
    bh = max(by2-by1,1e-6)

    area_a = aw*ah
    area_b = bw*bh

    dx = max(
        ax1-bx2,
        bx1-ax2,
        0
    )

    dy = max(
        ay1-by2,
        by1-ay2,
        0
    )

    gap = math.hypot(dx,dy)

    return (
        gap /
        math.sqrt(max(area_a,area_b))
    )


# ============================================================
# Frame scores
# ============================================================

rows = []

for _,r in PAIR.iterrows():

    case = r["case"]

    if case not in completed_cases:
        continue

    frame = int(r["frame"])
    raw = float(r[SCORE_COL])

    if raw <= 0:

        gap = 0.0
        sep = 0.0

    else:

        ka = (
            case,
            frame,
            int(r["rank_a"])
        )

        kb = (
            case,
            frame,
            int(r["rank_b"])
        )

        if ka not in cand_lookup or kb not in cand_lookup:
            continue

        a = cand_lookup[ka]
        b = cand_lookup[kb]

        gap = edge_gap_norm(a,b)

        # Frozen development definition:
        # evidence must establish spatial independence.
        sep = raw if gap > 0 else 0.0

    rows.append({
        "case":case,
        "frame":frame,
        "raw_score":raw,
        "edge_gap_norm":gap,
        "separated_score":sep
    })


frame = pd.DataFrame(rows)

frame.to_csv(
    "IDENTITY_EVAL_PARTIAL56_FRAME_SCORES.csv",
    index=False
)


# ============================================================
# Video scores
# ============================================================

video_rows = []

for _,m in MANIFEST.iterrows():

    case = m["case"]

    q = frame[
        frame["case"] == case
    ]

    if q.empty:

        raw_peak = 0.0
        score = 0.0
        peak_frame = -1
        gap = 0.0

    else:

        raw_peak = float(
            q["raw_score"].max()
        )

        idx = q[
            "separated_score"
        ].idxmax()

        best = q.loc[idx]

        score = float(
            best["separated_score"]
        )

        peak_frame = int(
            best["frame"]
        )

        gap = float(
            best["edge_gap_norm"]
        )

    video_rows.append({
        "case":case,
        "label":int(m["label"]),
        "eval_split":m["eval_split"],
        "dataset":m["dataset"],
        "object_prompt":m["object_prompt"],
        "raw_peak":raw_peak,
        "identity_dup_score":score,
        "peak_frame":peak_frame,
        "edge_gap_norm_at_peak":gap
    })


video = pd.DataFrame(video_rows)

video.to_csv(
    "IDENTITY_EVAL_PARTIAL56_VIDEO_SCORES.csv",
    index=False
)


# ============================================================
# Metrics
# ============================================================

def report(name,d):

    y = d["label"].values
    s = d["identity_dup_score"].values

    auc = roc_auc_score(y,s)
    ap = average_precision_score(y,s)

    pos = d[d.label == 1]
    neg = d[d.label == 0]

    print("\n" + "="*100)
    print(name)
    print("="*100)

    print(
        f"N={len(d)} "
        f"POS={len(pos)} "
        f"NEG={len(neg)}"
    )

    print(f"AUROC = {auc:.6f}")
    print(f"AP    = {ap:.6f}")

    print(
        f"Lowest positive = "
        f"{pos.identity_dup_score.min():.6f}"
    )

    print(
        f"Highest negative = "
        f"{neg.identity_dup_score.max():.6f}"
    )

    print("\nLOWEST POSITIVES")

    print(
        pos.sort_values(
            "identity_dup_score"
        )[
            [
                "case",
                "eval_split",
                "identity_dup_score",
                "peak_frame"
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print("\nHIGHEST NEGATIVES")

    print(
        neg.sort_values(
            "identity_dup_score",
            ascending=False
        )[
            [
                "case",
                "eval_split",
                "identity_dup_score",
                "peak_frame"
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    return auc,ap


pooled_auc, pooled_ap = report(
    "PARTIAL POOLED EVALUATION",
    video
)


heldout = video[
    video["eval_split"].isin(
        [
            "HELDOUT_POS",
            "HELDOUT_NEG"
        ]
    )
]

held_auc, held_ap = report(
    "PARTIAL HELD-OUT EVALUATION",
    heldout
)


newseed = heldout[
    heldout["dataset"].isin(
        [
            "Cosmos3_seed102",
            "Cosmos3_seed103"
        ]
    )
]

if newseed["label"].nunique() == 2:

    report(
        "PARTIAL HELD-OUT COSMOS3 SEED102/103",
        newseed
    )


print("\n" + "="*100)
print("PRELIMINARY RESULT")
print("="*100)

print(
    f"POOLED  AUROC = {pooled_auc:.6f}"
)

print(
    f"HELDOUT AUROC = {held_auc:.6f}"
)

print(
    "\nNOTE: Partial evaluation only; "
    "5/61 cases excluded because SAM3 "
    "did not complete under current GPU availability."
)

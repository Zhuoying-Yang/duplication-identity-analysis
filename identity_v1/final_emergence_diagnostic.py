import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

manifest = pd.read_csv(
    "IDENTITY_EVAL_V1_REAUDITED_COMPLETED.csv"
)

pairs = pd.read_csv(
    "FINAL_SIZE_SPATIAL_PAIR_SCORES.csv"
)

sam = pd.read_csv(
    "sam_identity_eval_v1/"
    "SAM_IDENTITY_EVAL_V1_CANDIDATES.csv"
)


# ============================================================
# Per-frame maximum size+spatial evidence
# ============================================================

frame = (
    pairs.groupby(
        ["case", "frame"],
        as_index=False
    )["size_spatial"]
    .max()
)

# Include sampled frames with no valid pair as zero
grid = (
    sam[["case", "frame"]]
    .drop_duplicates()
    .merge(
        frame,
        on=["case", "frame"],
        how="left"
    )
    .sort_values(["case", "frame"])
)

grid["size_spatial"] = (
    grid["size_spatial"].fillna(0.0)
)


# ============================================================
# Video-level emergence features
#
# No tracking IDs.
# Baseline uses ONLY the first sampled frames.
# ============================================================

rows = []

for _, m in manifest.iterrows():

    case = m["case"]

    q = (
        grid[grid.case == case]
        .sort_values("frame")
        .reset_index(drop=True)
    )

    x = q["size_spatial"].to_numpy()

    if len(x) == 0:
        continue

    peak = float(np.max(x))

    r = {
        "case": case,
        "label": int(m["label"]),
        "dataset": m["dataset"],
        "eval_split": m["eval_split"],
        "revised_category": m.get(
            "revised_category", ""
        ),
        "peak": peak,
    }

    # Diagnostic only:
    # compare several early-baseline windows.
    for k in [3, 5, 7, 10]:

        kk = min(k, len(x))

        early = x[:kk]

        baseline_max = float(
            np.max(early)
        )

        baseline_median = float(
            np.median(early)
        )

        later = x[kk:]

        later_peak = (
            float(np.max(later))
            if len(later)
            else peak
        )

        # Conservative:
        # must exceed anything seen in early baseline.
        r[f"delta_max_{k}"] = max(
            0.0,
            later_peak - baseline_max
        )

        # Softer version
        r[f"delta_median_{k}"] = max(
            0.0,
            later_peak - baseline_median
        )

        # Relative increase, stabilized
        r[f"ratio_{k}"] = (
            later_peak /
            (baseline_max + 0.05)
        )

    rows.append(r)


d = pd.DataFrame(rows)

d.to_csv(
    "FINAL_EMERGENCE_DIAGNOSTIC.csv",
    index=False
)


# ============================================================
# Evaluation
# ============================================================

scores = ["peak"]

for k in [3,5,7,10]:
    scores += [
        f"delta_max_{k}",
        f"delta_median_{k}",
        f"ratio_{k}",
    ]


def report(name, q):

    print("\n" + "="*105)
    print(name)
    print("="*105)

    print(
        f"N={len(q)} "
        f"POS={(q.label==1).sum()} "
        f"NEG={(q.label==0).sum()}"
    )

    for s in scores:

        auc = roc_auc_score(
            q.label,
            q[s]
        )

        ap = average_precision_score(
            q.label,
            q[s]
        )

        print(
            f"{s:20s} "
            f"AUROC={auc:.4f} "
            f"AP={ap:.4f}"
        )


report(
    "ALL 54 REAUDITED",
    d
)

new = d[
    d.dataset.isin(
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


print("\n" + "="*120)
print("SEED102/103 CASES — BASELINE EMERGENCE")
print("="*120)

print(
    new[
        [
            "case",
            "label",
            "peak",
            "delta_max_3",
            "delta_max_5",
            "delta_max_7",
            "delta_max_10",
        ]
    ]
    .sort_values(
        ["label","delta_max_5"],
        ascending=[False,False]
    )
    .to_string(index=False)
)

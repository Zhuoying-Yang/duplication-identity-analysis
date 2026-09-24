import math
from pathlib import Path
import pandas as pd

# ------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------

POS_PAIR = Path(
    "persistent_pair_diagnostic/PAIR_DIAGNOSTIC.csv"
)
NEG_PAIR = Path(
    "persistent_pair_negative6/PAIR_DIAGNOSTIC.csv"
)

POS_CAND = Path(
    "sam_masked_expansion6/SAM_EXPANSION6_CANDIDATES.csv"
)
NEG_CAND = Path(
    "sam_masked_negative6/SAM_NEGATIVE6_CANDIDATES.csv"
)

OUT = Path("fragmentation_geometry_diagnostic")
OUT.mkdir(exist_ok=True)

# Human-audited true duplication frames.
# Use frames where the second instance is clearly visible,
# not ambiguous pre-onset frames.
TRUE_FRAMES = {
    "Cosmos2.5_0058": [27, 45, 87],
    "Cosmos2.5_0060": [18, 27, 45, 90],
    "Cosmos3_seed101_0020": [141, 144, 174, 186],
    "Cosmos3_seed101_0036": [132, 135, 171, 183],
    "LVP_0030": [45, 48],
}

# Human-audited false duplication:
# one brown cube fragmented into pieces.
FALSE_FRAMES = {
    "Cosmos3_seed101_0039": [174, 177, 180, 183, 186, 188],
}


def load_best_pairs(pair_path, frame_dict, label):
    d = pd.read_csv(pair_path)

    rows = []

    for case, frames in frame_dict.items():
        for frame in frames:
            q = d[
                (d["case"] == case) &
                (d["frame"] == frame)
            ].copy()

            if q.empty:
                print("MISSING PAIR:", case, frame)
                continue

            # Current metric's strongest pair at this frame.
            if "score" in q.columns:
                score_col = "score"
            elif "eligible_dup_evidence" in q.columns:
                score_col = "eligible_dup_evidence"
            elif "dup_evidence" in q.columns:
                score_col = "dup_evidence"
            else:
                raise RuntimeError(
                    f"No score column found. Columns: {q.columns.tolist()}"
                )

            q = q.sort_values(score_col, ascending=False)
            r = q.iloc[0].copy()
            r["score"] = r[score_col]

            r["audit_label"] = label
            rows.append(r)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


pos = load_best_pairs(
    POS_PAIR,
    TRUE_FRAMES,
    "TRUE_DUP",
)

neg = load_best_pairs(
    NEG_PAIR,
    FALSE_FRAMES,
    "FRAGMENT_FALSE",
)

pairs = pd.concat([pos, neg], ignore_index=True)

pc = pd.read_csv(POS_CAND)
nc = pd.read_csv(NEG_CAND)

candidates = pd.concat([pc, nc], ignore_index=True)


def get_candidate(case, frame, rank):
    q = candidates[
        (candidates["case"] == case) &
        (candidates["frame"] == frame) &
        (candidates["rank"] == rank)
    ]

    if q.empty:
        raise RuntimeError(
            f"candidate missing: {case} f{frame} r{rank}"
        )

    return q.iloc[0]


def geometry(a, b):
    ax1, ay1, ax2, ay2 = [
        float(a[x]) for x in ["x1","y1","x2","y2"]
    ]
    bx1, by1, bx2, by2 = [
        float(b[x]) for x in ["x1","y1","x2","y2"]
    ]

    aw = max(ax2-ax1, 1e-6)
    ah = max(ay2-ay1, 1e-6)
    bw = max(bx2-bx1, 1e-6)
    bh = max(by2-by1, 1e-6)

    area_a = aw * ah
    area_b = bw * bh

    acx = (ax1+ax2)/2
    acy = (ay1+ay2)/2
    bcx = (bx1+bx2)/2
    bcy = (by1+by2)/2

    center_dist = math.hypot(acx-bcx, acy-bcy)

    # Distance between closest edges.
    dx = max(ax1-bx2, bx1-ax2, 0)
    dy = max(ay1-by2, by1-ay2, 0)
    edge_gap = math.hypot(dx,dy)

    max_area = max(area_a, area_b)
    max_diag = max(
        math.hypot(aw,ah),
        math.hypot(bw,bh)
    )

    # Enclosing rectangle of both candidates.
    ex1 = min(ax1,bx1)
    ey1 = min(ay1,by1)
    ex2 = max(ax2,bx2)
    ey2 = max(ay2,by2)

    enclosing_area = max(
        (ex2-ex1)*(ey2-ey1),
        1e-6
    )

    return {
        "box_area_a": area_a,
        "box_area_b": area_b,

        # If one is just a tiny fragment, this becomes small.
        "box_size_ratio":
            min(area_a,area_b) / max_area,

        # Absolute center separation.
        "center_distance":
            center_dist,

        # Main scale-normalized separation measures.
        "center_norm_sqrt_area":
            center_dist / math.sqrt(max_area),

        "center_norm_diag":
            center_dist / max_diag,

        # Actual empty spatial gap between boxes.
        "edge_gap":
            edge_gap,

        "edge_gap_norm":
            edge_gap / math.sqrt(max_area),

        # How large a spatial region is needed to contain both.
        "enclosing_ratio":
            enclosing_area / max_area,

        # Adjacent fragments tend to pack tightly into their
        # joint enclosing region.
        "pair_fill":
            (area_a + area_b) / enclosing_area,

        "mask_size_ratio":
            min(float(a["mask_pixels"]),
                float(b["mask_pixels"]))
            /
            max(float(a["mask_pixels"]),
                float(b["mask_pixels"]),
                1.0),

        "mask_density_a":
            float(a["mask_pixels"]) / area_a,

        "mask_density_b":
            float(b["mask_pixels"]) / area_b,
    }


rows = []

for _, r in pairs.iterrows():

    a = get_candidate(
        r["case"],
        int(r["frame"]),
        int(r["rank_a"]),
    )

    b = get_candidate(
        r["case"],
        int(r["frame"]),
        int(r["rank_b"]),
    )

    g = geometry(a,b)

    row = {
        "audit_label": r["audit_label"],
        "case": r["case"],
        "frame": int(r["frame"]),

        "rank_a": int(r["rank_a"]),
        "rank_b": int(r["rank_b"]),

        "dup_score": float(r["score"]),
        "dino_similarity":
            float(r["pair_similarity"]),

        "min_siglip":
            float(r["min_siglip"]),

        "min_gdino":
            float(r["min_gdino"]),

        "iou":
            float(r["iou"])
            if "iou" in r and pd.notna(r["iou"])
            else float("nan"),

        "containment":
            float(r["containment"])
            if "containment" in r
            and pd.notna(r["containment"])
            else float("nan"),

        **g,
    }

    rows.append(row)


out = pd.DataFrame(rows)

csv = OUT / "GEOMETRY_AUDIT.csv"
out.to_csv(csv, index=False)

show_cols = [
    "audit_label",
    "case",
    "frame",
    "dup_score",
    "dino_similarity",
    "min_gdino",
    "box_size_ratio",
    "center_norm_sqrt_area",
    "center_norm_diag",
    "edge_gap_norm",
    "enclosing_ratio",
    "pair_fill",
    "mask_size_ratio",
]

print("\n" + "="*150)
print("FRAME-LEVEL GEOMETRY")
print("="*150)

print(
    out[show_cols]
    .sort_values(
        ["audit_label","case","frame"]
    )
    .to_string(index=False)
)

print("\n" + "="*150)
print("GROUP SUMMARY — MEDIAN")
print("="*150)

metrics = [
    "dup_score",
    "dino_similarity",
    "box_size_ratio",
    "center_norm_sqrt_area",
    "center_norm_diag",
    "edge_gap_norm",
    "enclosing_ratio",
    "pair_fill",
    "mask_size_ratio",
]

print(
    out.groupby("audit_label")[metrics]
       .median()
       .T
       .to_string()
)

print("\n" + "="*150)
print("GROUP RANGE [MIN, MAX]")
print("="*150)

for metric in metrics:
    print("\n", metric)

    for label, q in out.groupby("audit_label"):
        print(
            f"{label:15s} "
            f"min={q[metric].min():.4f} "
            f"median={q[metric].median():.4f} "
            f"max={q[metric].max():.4f}"
        )

print("\nCSV:", csv)

from pathlib import Path
import itertools

import numpy as np
import pandas as pd
import torch
from PIL import Image

from transformers import AutoImageProcessor, Dinov2Model


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/shared/ssd_30T/zhuoyingyang/physact"
)

BASE = (
    ROOT / "sam3_robowm/identity_v1"
)

IN_CSV = (
    BASE
    / "sam_identity_eval_v1"
    / "SAM_IDENTITY_EVAL_V1_CANDIDATES.csv"
)

OUT_DIR = (
    BASE
    / "dinov2_identity_eval_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PAIR_CSV = (
    OUT_DIR
    / "DINOV2_IDENTITY_EVAL_V1_PAIRS.csv"
)

FRAME_CSV = (
    OUT_DIR
    / "DINOV2_IDENTITY_EVAL_V1_FRAMES.csv"
)

CASE_CSV = (
    OUT_DIR
    / "DINOV2_IDENTITY_EVAL_V1_CASE_SUMMARY.csv"
)


# ============================================================
# FROZEN SETTINGS
# ============================================================

MODEL_NAME = "facebook/dinov2-base"

CACHE = (
    "/shared/ssd_30T/zhuoyingyang/models/hf_cache"
)

# Frozen from development pilot.
CONTAINMENT_REJECT = 0.90


# ============================================================
# GEOMETRY
# ============================================================

def geometry(a, b):

    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])

    iw = max(
        0.0,
        ix2 - ix1
    )

    ih = max(
        0.0,
        iy2 - iy1
    )

    inter = iw * ih

    area_a = max(
        1.0,
        (a["x2"] - a["x1"])
        * (a["y2"] - a["y1"])
    )

    area_b = max(
        1.0,
        (b["x2"] - b["x1"])
        * (b["y2"] - b["y1"])
    )

    union = (
        area_a
        + area_b
        - inter
    )

    iou = (
        inter / union
        if union > 0
        else 0.0
    )

    containment = (
        inter
        / min(area_a, area_b)
    )

    return iou, containment


# ============================================================
# MODEL
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 100)
print("DINOv2 EXPANSION-6 FROZEN")
print("=" * 100)

print("device:", device)

processor = (
    AutoImageProcessor
    .from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE,
        local_files_only=True,
    )
)

model = (
    Dinov2Model
    .from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE,
        local_files_only=True,
    )
    .to(device)
    .eval()
)

print("DINOv2 ready.")


@torch.no_grad()
def embed(images):

    x = processor(
        images=images,
        return_tensors="pt",
    )

    out = model(
        pixel_values=
            x["pixel_values"].to(device)
    )

    # Same CLS representation as development pilot.
    f = (
        out.last_hidden_state[:, 0]
    )

    f = (
        f
        / (
            f.norm(
                dim=-1,
                keepdim=True
            )
            + 1e-8
        )
    )

    return f


# ============================================================
# LOAD
# ============================================================

d = pd.read_csv(
    IN_CSV
)

print(
    "candidate rows:",
    len(d)
)

print(
    "cases:",
    d["case"].nunique()
)


# ============================================================
# RUN
# ============================================================

pair_rows = []
frame_rows = []


for case in d["case"].unique():

    print()
    print("=" * 100)
    print(case)
    print("=" * 100)

    z = (
        d[d["case"] == case]
        .copy()
    )

    frames = sorted(
        z["frame"]
        .astype(int)
        .unique()
    )

    for frame in frames:

        g = (
            z[z["frame"] == frame]
            .sort_values("rank")
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # Fewer than 2 candidates = no duplication-pair
        # evidence in this sampled frame.
        # ----------------------------------------------------

        if len(g) < 2:

            frame_rows.append({
                "case": case,
                "frame": int(frame),
                "n_crops": len(g),
                "n_pairs": 0,
                "n_distinct_pairs": 0,
                "frame_score": 0.0,
                "best_pair": "",
                "best_containment":
                    np.nan,
                "best_iou":
                    np.nan,
            })

            print(
                f"frame={int(frame):03d} "
                f"N={len(g)} "
                f"score=0.000 "
                f"(no pair)"
            )

            continue

        # ----------------------------------------------------
        # DINO embeddings for all available SAM crops
        # ----------------------------------------------------

        images = [
            Image.open(
                p
            ).convert("RGB")
            for p in g["crop_path"]
        ]

        feats = embed(
            images
        )

        sim_matrix = (
            feats @ feats.T
        ).detach().cpu().numpy()

        local_pairs = []

        for i, j in itertools.combinations(
            range(len(g)),
            2
        ):

            a = g.iloc[i]
            b = g.iloc[j]

            iou, containment = geometry(
                a,
                b
            )

            pair_sim = float(
                sim_matrix[i, j]
            )

            nested = (
                containment
                >= CONTAINMENT_REJECT
            )

            row = {
                "case": case,
                "frame": int(frame),

                "rank_a":
                    int(a["rank"]),

                "rank_b":
                    int(b["rank"]),

                "pair_similarity":
                    pair_sim,

                "iou":
                    iou,

                "containment":
                    containment,

                "nested":
                    nested,

                "gdino_a":
                    float(
                        a["gdino_score"]
                    ),

                "gdino_b":
                    float(
                        b["gdino_score"]
                    ),

                "siglip_source_a":
                    float(
                        a["source_similarity"]
                    ),

                "siglip_source_b":
                    float(
                        b["source_similarity"]
                    ),

                "sam_a":
                    float(
                        a["sam_score"]
                    ),

                "sam_b":
                    float(
                        b["sam_score"]
                    ),

                "crop_a":
                    a["crop_path"],

                "crop_b":
                    b["crop_path"],
            }

            pair_rows.append(
                row
            )

            local_pairs.append(
                row
            )

        # ----------------------------------------------------
        # Frozen frame score:
        # max DINOv2 pair similarity among non-nested pairs.
        # ----------------------------------------------------

        valid = [
            r
            for r in local_pairs
            if not r["nested"]
        ]

        if not valid:

            score = 0.0
            best_pair = ""
            best_cont = np.nan
            best_iou = np.nan

        else:

            best = max(
                valid,
                key=lambda r:
                    r["pair_similarity"]
            )

            score = float(
                best["pair_similarity"]
            )

            best_pair = (
                f"{best['rank_a']}"
                f"-"
                f"{best['rank_b']}"
            )

            best_cont = float(
                best["containment"]
            )

            best_iou = float(
                best["iou"]
            )

        frame_rows.append({
            "case": case,
            "frame": int(frame),

            "n_crops":
                len(g),

            "n_pairs":
                len(local_pairs),

            "n_distinct_pairs":
                len(valid),

            "frame_score":
                score,

            "best_pair":
                best_pair,

            "best_containment":
                best_cont,

            "best_iou":
                best_iou,
        })

        print(
            f"frame={int(frame):03d} "
            f"N={len(g)} "
            f"distinct={len(valid)} "
            f"score={score:.3f} "
            f"best={best_pair}"
        )


# ============================================================
# SAVE RAW PAIR + FRAME RESULTS
# ============================================================

pairs = pd.DataFrame(
    pair_rows
)

frames = pd.DataFrame(
    frame_rows
)

pairs.to_csv(
    PAIR_CSV,
    index=False
)

frames.to_csv(
    FRAME_CSV,
    index=False
)


# ============================================================
# CASE SUMMARY
#
# NO threshold.
# NO temporal smoothing.
# Just report raw trajectory statistics and peaks.
# ============================================================

case_rows = []

print()
print("=" * 100)
print("CASE SUMMARY")
print("=" * 100)


for case, g in frames.groupby("case"):

    g = g.sort_values(
        "frame"
    )

    top = (
        g.sort_values(
            "frame_score",
            ascending=False
        )
        .head(10)
    )

    peak = top.iloc[0]

    case_rows.append({
        "case": case,

        "n_frames":
            len(g),

        "peak_frame":
            int(
                peak["frame"]
            ),

        "peak_score":
            float(
                peak["frame_score"]
            ),

        "median_score":
            float(
                g["frame_score"]
                .median()
            ),

        "mean_score":
            float(
                g["frame_score"]
                .mean()
            ),

        "p90_score":
            float(
                g["frame_score"]
                .quantile(0.90)
            ),

        "frames_with_pair":
            int(
                (
                    g["n_distinct_pairs"]
                    > 0
                ).sum()
            ),
    })

    print()
    print(case)

    print(
        "peak =",
        round(
            float(
                peak["frame_score"]
            ),
            3
        ),
        "@ frame",
        int(
            peak["frame"]
        )
    )

    print("Top 10 frames:")

    print(
        top[
            [
                "frame",
                "frame_score",
                "best_pair",
                "n_crops",
                "n_distinct_pairs",
            ]
        ].to_string(
            index=False
        )
    )


case_summary = pd.DataFrame(
    case_rows
)

case_summary.to_csv(
    CASE_CSV,
    index=False
)


print()
print("=" * 100)
print("FINAL CASE TABLE")
print("=" * 100)

print(
    case_summary
    .sort_values(
        "peak_score",
        ascending=False
    )
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("DONE")
print("=" * 100)

print("PAIRS :", PAIR_CSV)
print("FRAMES:", FRAME_CSV)
print("CASES :", CASE_CSV)

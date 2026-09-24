from pathlib import Path
import itertools

import numpy as np
import pandas as pd
import torch
from PIL import Image

from transformers import AutoImageProcessor, Dinov2Model


ROOT = Path("/shared/ssd_30T/zhuoyingyang/physact")
BASE = ROOT / "sam3_robowm/identity_v1"

SAM_BASE = BASE / "sam_masked_identity_pilot"

META_CSV = SAM_BASE / "SAM_MASKED_CANDIDATES.csv"
GDINO_CSV = (
    BASE
    / "siglip_source_match_pilot"
    / "SIGLIP_CANDIDATES.csv"
)

SOURCE_DIR = SAM_BASE / "source"

OUT_PAIRS = BASE / "DINOV2_MASKED_PAIR_PILOT.csv"
OUT_FRAMES = BASE / "DINOV2_MASKED_FRAME_PILOT.csv"

MODEL_NAME = "facebook/dinov2-base"
CACHE = "/shared/ssd_30T/zhuoyingyang/models/hf_cache"


FRAME_GT = {
    ("Cosmos2.5_0016", 33): "TRUE_DUP",
    ("Cosmos2.5_0016", 60): "TRUE_DUP",

    ("Cosmos3_seed101_0003", 144): "TRUE_DUP",
    ("Cosmos3_seed101_0003", 156): "TRUE_DUP",
    ("Cosmos3_seed101_0003", 162): "TRUE_DUP",
    ("Cosmos3_seed101_0003", 168): "TRUE_DUP",

    ("Cosmos3_seed101_0003", 30): "NO_DUP",
    ("Cosmos3_seed101_0003", 54): "NO_DUP",
    ("Cosmos3_seed101_0003", 99): "NO_DUP",
    ("Cosmos3_seed101_0003", 123): "NO_DUP",

    ("LVP_0004", 12): "NO_DUP",
    ("LVP_0004", 15): "NO_DUP",
    ("LVP_0004", 18): "NO_DUP",
    ("LVP_0004", 30): "NO_DUP",
}


def geometry(a, b):
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

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

    union = area_a + area_b - inter

    iou = inter / union
    containment = inter / min(area_a, area_b)

    return iou, containment


device = "cuda" if torch.cuda.is_available() else "cpu"

print("device:", device)
print("Loading DINOv2...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE,
    local_files_only=True,
)

model = Dinov2Model.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE,
    local_files_only=True,
).to(device).eval()

print("DINOv2 ready.")


@torch.no_grad()
def embed(images):
    x = processor(
        images=images,
        return_tensors="pt",
    )

    pixel_values = x["pixel_values"].to(device)

    out = model(
        pixel_values=pixel_values
    )

    # DINOv2 CLS token
    f = out.last_hidden_state[:, 0]

    f = f / (
        f.norm(
            dim=-1,
            keepdim=True
        ) + 1e-8
    )

    return f


meta = pd.read_csv(META_CSV)
gdino = pd.read_csv(GDINO_CSV)


# Recreate rank -> original GDINO bbox lookup.
box_lookup = {}

for (case, frame), g in gdino.groupby(
    ["case", "frame"]
):
    g = (
        g.sort_values(
            "source_similarity",
            ascending=False
        )
        .head(3)
        .reset_index(drop=True)
    )

    for i, r in g.iterrows():
        box_lookup[
            (
                case,
                int(frame),
                i + 1,
            )
        ] = {
            "x1": float(r.x1),
            "y1": float(r.y1),
            "x2": float(r.x2),
            "y2": float(r.y2),
        }


pair_rows = []


for (case, frame), gt in FRAME_GT.items():

    print("\n" + "=" * 90)
    print(case, "| frame", frame, "|", gt)
    print("=" * 90)

    source_path = SOURCE_DIR / f"{case}.png"

    if not source_path.exists():
        print("MISSING SOURCE:", source_path)
        continue

    source = Image.open(
        source_path
    ).convert("RGB")

    src_feat = embed([source])[0]

    g = meta[
        (meta.case == case)
        & (meta.frame == frame)
    ].copy()

    g = (
        g.sort_values("rank")
        .reset_index(drop=True)
    )

    if len(g) < 2:
        print("SKIP: fewer than 2 candidates")
        continue

    images = [
        Image.open(p).convert("RGB")
        for p in g.crop_path
    ]

    feats = embed(images)

    src_sims = (
        feats @ src_feat
    ).detach().cpu().numpy()

    pair_matrix = (
        feats @ feats.T
    ).detach().cpu().numpy()

    for i, j in itertools.combinations(
        range(len(g)),
        2
    ):

        a = g.iloc[i]
        b = g.iloc[j]

        rank_a = int(a["rank"])
        rank_b = int(b["rank"])

        key_a = (case, frame, rank_a)
        key_b = (case, frame, rank_b)

        if (
            key_a not in box_lookup
            or key_b not in box_lookup
        ):
            continue

        iou, containment = geometry(
            box_lookup[key_a],
            box_lookup[key_b],
        )

        sim_a = float(src_sims[i])
        sim_b = float(src_sims[j])

        pair_sim = float(
            pair_matrix[i, j]
        )

        min_source = min(
            sim_a,
            sim_b
        )

        raw_score = (
            min_source
            * max(0.0, pair_sim)
        )

        # Only remove near-complete nested detections.
        nested = containment >= 0.90

        distinct_score = (
            0.0
            if nested
            else raw_score
        )

        print(
            f"R{rank_a}-R{rank_b} "
            f"src=({sim_a:.3f},{sim_b:.3f}) "
            f"pair={pair_sim:.3f} "
            f"IoU={iou:.3f} "
            f"cont={containment:.3f} "
            f"nested={nested} "
            f"score={distinct_score:.3f}"
        )

        pair_rows.append({
            "case": case,
            "frame": frame,
            "gt": gt,

            "rank_a": rank_a,
            "rank_b": rank_b,

            "src_sim_a": sim_a,
            "src_sim_b": sim_b,
            "min_source_sim": min_source,

            "pair_similarity": pair_sim,

            "iou": iou,
            "containment": containment,
            "nested": nested,

            "raw_pair_score": raw_score,
            "distinct_pair_score": distinct_score,
        })


pairs = pd.DataFrame(pair_rows)

pairs.to_csv(
    OUT_PAIRS,
    index=False
)


# ============================================================
# FRAME LEVEL: best non-nested pair
# ============================================================

frame_rows = []

for (
    case,
    frame,
    gt
), g in pairs.groupby(
    ["case", "frame", "gt"]
):

    best = (
        g.sort_values(
            "distinct_pair_score",
            ascending=False
        )
        .iloc[0]
    )

    frame_rows.append({
        "case": case,
        "frame": frame,
        "gt": gt,

        "dinov2_frame_score":
            float(
                best.distinct_pair_score
            ),

        "best_pair":
            f"{int(best.rank_a)}-{int(best.rank_b)}",

        "src_sim_a":
            float(best.src_sim_a),

        "src_sim_b":
            float(best.src_sim_b),

        "pair_similarity":
            float(best.pair_similarity),

        "containment":
            float(best.containment),

        "nested":
            bool(best.nested),
    })


frames = pd.DataFrame(frame_rows)

frames.to_csv(
    OUT_FRAMES,
    index=False
)


print("\n" + "=" * 90)
print("FRAME-LEVEL SUMMARY")
print("=" * 90)

print(
    frames.sort_values(
        [
            "gt",
            "dinov2_frame_score",
        ],
        ascending=[
            True,
            False,
        ],
    ).to_string(index=False)
)


print("\n" + "=" * 90)
print("GROUP SUMMARY")
print("=" * 90)

for gt, g in frames.groupby("gt"):
    print(
        gt,
        "| N =", len(g),
        "| mean =",
        round(
            g.dinov2_frame_score.mean(),
            3
        ),
        "| median =",
        round(
            g.dinov2_frame_score.median(),
            3
        ),
        "| min =",
        round(
            g.dinov2_frame_score.min(),
            3
        ),
        "| max =",
        round(
            g.dinov2_frame_score.max(),
            3
        ),
    )


print("\nSaved:")
print(OUT_PAIRS)
print(OUT_FRAMES)

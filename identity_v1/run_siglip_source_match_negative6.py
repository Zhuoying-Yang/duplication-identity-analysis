from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from groundingdino.util.inference import (
    load_model,
    load_image,
    predict,
)

from transformers import (
    AutoImageProcessor,
    SiglipModel,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/shared/ssd_30T/zhuoyingyang/physact"
)

IDENTITY = (
    ROOT / "sam3_robowm/identity_v1"
)

MANIFEST = (
    IDENTITY / "NEGATIVE6_FROZEN.csv"
)

PROPOSALS = (
    IDENTITY
    / "gdino_negative6_frozen"
    / "GDINO_NEGATIVE6_PROPOSALS.csv"
)

OUT = (
    IDENTITY
    / "siglip_source_match_negative6"
)

DEBUG = OUT / "debug"
SOURCE_OUT = OUT / "source_targets"

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

DEBUG.mkdir(
    parents=True,
    exist_ok=True,
)

SOURCE_OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# GROUNDINGDINO
# ============================================================

GDINO = (
    ROOT
    / "ReVidgen/pkgs/"
    "Grounded-Segment-Anything/"
    "GroundingDINO"
)

CONFIG = (
    GDINO
    / "groundingdino/config/"
    "GroundingDINO_SwinB.py"
)

CHECKPOINT = (
    ROOT
    / "ReVidgen/checkpoints/"
    "GroundingDino/"
    "groundingdino_swinb_cogcoor.pth"
)

GDINO_BOX_THRESHOLD = 0.10
TEXT_THRESHOLD = 0.10
NMS_IOU = 0.50
MAX_PROPOSALS = 12

# Same crop expansion as original pilot
CROP_EXPAND = 0.08


# ============================================================
# SIGLIP
# ============================================================

SIGLIP_CACHE = (
    ROOT
    / "cosmos_predict25/hf_cache/hub/"
    "models--google--siglip-so400m-patch14-384"
)

snapshots = sorted(
    (SIGLIP_CACHE / "snapshots").glob("*")
)

if not snapshots:
    raise RuntimeError(
        f"No SigLIP snapshot found: {SIGLIP_CACHE}"
    )

SIGLIP_PATH = snapshots[-1]


# ============================================================
# HELPERS
# ============================================================

def cxcywh_to_xyxy(boxes, W, H):

    if len(boxes) == 0:
        return np.zeros(
            (0, 4),
            dtype=np.float32,
        )

    b = np.asarray(
        boxes,
        dtype=np.float32,
    )

    cx = b[:, 0] * W
    cy = b[:, 1] * H
    bw = b[:, 2] * W
    bh = b[:, 3] * H

    return np.stack(
        [
            cx - bw / 2,
            cy - bh / 2,
            cx + bw / 2,
            cy + bh / 2,
        ],
        axis=1,
    )


def iou_one_to_many(box, boxes):

    if len(boxes) == 0:
        return np.zeros(
            0,
            dtype=np.float32,
        )

    ix1 = np.maximum(
        box[0],
        boxes[:, 0],
    )

    iy1 = np.maximum(
        box[1],
        boxes[:, 1],
    )

    ix2 = np.minimum(
        box[2],
        boxes[:, 2],
    )

    iy2 = np.minimum(
        box[3],
        boxes[:, 3],
    )

    iw = np.maximum(
        0.0,
        ix2 - ix1,
    )

    ih = np.maximum(
        0.0,
        iy2 - iy1,
    )

    inter = iw * ih

    area_a = max(
        1e-6,
        (box[2] - box[0])
        * (box[3] - box[1]),
    )

    area_b = np.maximum(
        1e-6,
        (boxes[:, 2] - boxes[:, 0])
        * (boxes[:, 3] - boxes[:, 1]),
    )

    union = (
        area_a
        + area_b
        - inter
    )

    return (
        inter
        / np.maximum(
            union,
            1e-6,
        )
    )


def nms(boxes, scores):

    if len(boxes) == 0:
        return []

    boxes = np.asarray(boxes)
    scores = np.asarray(scores)

    order = np.argsort(
        scores
    )[::-1]

    keep = []

    while len(order) > 0:

        i = int(order[0])
        keep.append(i)

        if len(order) == 1:
            break

        rest = order[1:]

        ov = iou_one_to_many(
            boxes[i],
            boxes[rest],
        )

        order = rest[
            ov <= NMS_IOU
        ]

    return keep


def crop_box(
    image_rgb,
    box,
):

    H, W = image_rgb.shape[:2]

    x1, y1, x2, y2 = [
        float(v)
        for v in box
    ]

    bw = x2 - x1
    bh = y2 - y1

    x1 -= CROP_EXPAND * bw
    x2 += CROP_EXPAND * bw

    y1 -= CROP_EXPAND * bh
    y2 += CROP_EXPAND * bh

    x1 = max(
        0,
        int(x1),
    )

    y1 = max(
        0,
        int(y1),
    )

    x2 = min(
        W,
        int(x2),
    )

    y2 = min(
        H,
        int(y2),
    )

    if x2 <= x1 or y2 <= y1:
        return None

    return Image.fromarray(
        image_rgb[
            y1:y2,
            x1:x2
        ]
    )


def read_video_frame(
    cap,
    frame_idx,
):

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(frame_idx),
    )

    ok, bgr = cap.read()

    if not ok:
        return None

    return cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    )


def draw_debug(
    rgb,
    rows,
    case,
    frame,
):

    img = Image.fromarray(
        rgb
    ).convert("RGB")

    d = ImageDraw.Draw(img)

    colors = [
        "red",
        "blue",
        "lime",
    ]

    for k, r in enumerate(
        rows[:3]
    ):

        color = colors[
            min(
                k,
                len(colors) - 1
            )
        ]

        d.rectangle(
            [
                r["x1"],
                r["y1"],
                r["x2"],
                r["y2"],
            ],
            outline=color,
            width=4,
        )

        d.text(
            (
                r["x1"] + 4,
                r["y1"] + 4,
            ),
            (
                f"R{k+1} "
                f"S={r['source_similarity']:.2f} "
                f"G={r['gdino_score']:.2f}"
            ),
            fill=color,
            stroke_width=2,
            stroke_fill="black",
        )

    return img


# ============================================================
# LOAD DATA
# ============================================================

manifest = pd.read_csv(
    MANIFEST
)

proposal_df = pd.read_csv(
    PROPOSALS
)

print(
    "manifest cases:",
    len(manifest),
)

print(
    "proposal rows:",
    len(proposal_df),
)


# ============================================================
# LOAD MODELS
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 100)
print("SIGLIP SOURCE MATCH NEGATIVE-6")
print("=" * 100)

print("device:", device)
print("GDINO config:", CONFIG)
print("GDINO checkpoint:", CHECKPOINT)
print("SigLIP:", SIGLIP_PATH)

gdino_model = load_model(
    str(CONFIG),
    str(CHECKPOINT),
    device=device,
)

print("GroundingDINO ready.")

processor = (
    AutoImageProcessor
    .from_pretrained(
        str(SIGLIP_PATH),
        local_files_only=True,
    )
)

siglip = (
    SiglipModel
    .from_pretrained(
        str(SIGLIP_PATH),
        local_files_only=True,
    )
    .to(device)
    .eval()
)

print("SigLIP ready.")


@torch.no_grad()
def embed_images(images):

    x = processor(
        images=images,
        return_tensors="pt",
    )

    feats = (
        siglip
        .get_image_features(
            pixel_values=
                x["pixel_values"].to(
                    device
                )
        )
    )

    feats = (
        feats
        / (
            feats.norm(
                dim=-1,
                keepdim=True,
            )
            + 1e-8
        )
    )

    return feats


# ============================================================
# RUN
# ============================================================

candidate_rows = []
summary_rows = []


for _, cfg in manifest.iterrows():

    case = cfg["case"]
    prompt = cfg["object_prompt"]

    video_path = Path(
        cfg["video_path"]
    )

    source_path = Path(
        cfg["source_path"]
    )

    print()
    print("=" * 100)
    print(case)
    print("prompt:", prompt)
    print("=" * 100)

    # ========================================================
    # SOURCE TARGET
    # ========================================================

    source_rgb = np.asarray(
        Image.open(
            source_path
        ).convert("RGB")
    )

    Hs, Ws = source_rgb.shape[:2]

    image_source, image_tensor = (
        load_image(
            str(source_path)
        )
    )

    boxes, logits, phrases = predict(
        model=gdino_model,
        image=image_tensor,
        caption=prompt,
        box_threshold=GDINO_BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=device,
    )

    if torch.is_tensor(boxes):
        boxes = (
            boxes
            .detach()
            .cpu()
            .numpy()
        )

    if torch.is_tensor(logits):
        logits = (
            logits
            .detach()
            .cpu()
            .numpy()
        )

    boxes = np.asarray(boxes)

    logits = (
        np.asarray(logits)
        .reshape(-1)
    )

    src_boxes = cxcywh_to_xyxy(
        boxes,
        Ws,
        Hs,
    )

    keep = nms(
        src_boxes,
        logits,
    )

    keep = keep[
        :MAX_PROPOSALS
    ]

    if not keep:
        print(
            "ERROR: no source target detection"
        )
        continue

    # Highest-GDINO source detection
    best = keep[0]

    source_crop = crop_box(
        source_rgb,
        src_boxes[best],
    )

    if source_crop is None:
        print(
            "ERROR: source crop failed"
        )
        continue

    source_crop_path = (
        SOURCE_OUT
        / f"{case}_SOURCE_TARGET.jpg"
    )

    source_crop.save(
        source_crop_path,
        quality=95,
    )

    source_feat = embed_images(
        [source_crop]
    )[0]

    print(
        "SOURCE bbox score =",
        round(
            float(logits[best]),
            3,
        )
    )

    print(
        "SOURCE crop:",
        source_crop_path,
    )

    # ========================================================
    # GENERATED VIDEO
    # ========================================================

    z = proposal_df[
        proposal_df["case"] == case
    ].copy()

    frames = sorted(
        z["frame"]
        .astype(int)
        .unique()
        .tolist()
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    case_debug = (
        DEBUG / case
    )

    case_debug.mkdir(
        parents=True,
        exist_ok=True,
    )

    for frame_idx in frames:

        rgb = read_video_frame(
            cap,
            frame_idx,
        )

        if rgb is None:
            continue

        g = z[
            z["frame"] == frame_idx
        ].copy()

        # Already frozen as GDINO 0.10 + NMS + top12
        g = (
            g.sort_values(
                "gdino_score",
                ascending=False,
            )
            .head(
                MAX_PROPOSALS
            )
            .reset_index(
                drop=True
            )
        )

        crops = []
        valid_rows = []

        for _, r in g.iterrows():

            box = [
                r["x1"],
                r["y1"],
                r["x2"],
                r["y2"],
            ]

            crop = crop_box(
                rgb,
                box,
            )

            if crop is None:
                continue

            crops.append(
                crop
            )

            valid_rows.append(
                r
            )

        if not crops:

            summary_rows.append({
                "case": case,
                "frame": frame_idx,
                "n_candidates": 0,
                "top1_similarity":
                    np.nan,
                "top2_similarity":
                    np.nan,
                "top3_similarity":
                    np.nan,
            })

            continue

        feats = embed_images(
            crops
        )

        sims = (
            feats
            @ source_feat
        ).detach().cpu().numpy()

        frame_candidates = []

        for r, sim in zip(
            valid_rows,
            sims,
        ):

            frame_candidates.append({
                "case": case,
                "frame": int(
                    frame_idx
                ),
                "prompt": prompt,
                "gdino_score":
                    float(
                        r["gdino_score"]
                    ),
                "source_similarity":
                    float(sim),
                "x1": float(r["x1"]),
                "y1": float(r["y1"]),
                "x2": float(r["x2"]),
                "y2": float(r["y2"]),
            })

        # Frozen selection logic:
        # rank by source similarity
        frame_candidates = sorted(
            frame_candidates,
            key=lambda x:
                x[
                    "source_similarity"
                ],
            reverse=True,
        )

        for rank, r in enumerate(
            frame_candidates,
            start=1,
        ):

            r["rank"] = rank

            candidate_rows.append(
                r
            )

        vals = [
            r["source_similarity"]
            for r in frame_candidates
        ]

        summary_rows.append({
            "case": case,
            "frame": int(
                frame_idx
            ),
            "n_candidates":
                len(vals),
            "top1_similarity":
                vals[0]
                if len(vals) >= 1
                else np.nan,
            "top2_similarity":
                vals[1]
                if len(vals) >= 2
                else np.nan,
            "top3_similarity":
                vals[2]
                if len(vals) >= 3
                else np.nan,
        })

        # Save only periodic debug images
        # to avoid huge output.
        if (
            frame_idx % 15 == 0
            or frame_idx == frames[-1]
        ):

            vis = draw_debug(
                rgb,
                frame_candidates,
                case,
                frame_idx,
            )

            vis.save(
                case_debug
                / f"frame_{frame_idx:04d}.jpg",
                quality=92,
            )

        top1 = (
            vals[0]
            if len(vals) >= 1
            else np.nan
        )

        top2 = (
            vals[1]
            if len(vals) >= 2
            else np.nan
        )

        print(
            f"frame={frame_idx:03d} "
            f"N={len(vals):2d} "
            f"S1={top1:.3f} "
            f"S2={top2:.3f}"
            if len(vals) >= 2
            else
            f"frame={frame_idx:03d} "
            f"N={len(vals):2d} "
            f"S1={top1:.3f} "
            f"S2=nan"
        )

    cap.release()


# ============================================================
# SAVE
# ============================================================

candidate_csv = (
    OUT
    / "SIGLIP_NEGATIVE6_CANDIDATES.csv"
)

summary_csv = (
    OUT
    / "SIGLIP_NEGATIVE6_FRAME_SUMMARY.csv"
)

pd.DataFrame(
    candidate_rows
).to_csv(
    candidate_csv,
    index=False,
)

pd.DataFrame(
    summary_rows
).to_csv(
    summary_csv,
    index=False,
)

print()
print("=" * 100)
print("DONE")
print("=" * 100)

print(
    "CANDIDATES:",
    candidate_csv,
)

print(
    "SUMMARY:",
    summary_csv,
)

print(
    "SOURCE TARGETS:",
    SOURCE_OUT,
)

print(
    "DEBUG:",
    DEBUG,
)

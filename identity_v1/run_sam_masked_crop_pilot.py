from pathlib import Path
import csv
import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


ROOT = Path("/shared/ssd_30T/zhuoyingyang/physact")

IN_CSV = (
    ROOT / "sam3_robowm/identity_v1/"
    "siglip_source_match_pilot/SIGLIP_CANDIDATES.csv"
)

SOURCE_ROOT = (
    ROOT / "sam3_robowm/identity_v1/"
    "siglip_source_match_pilot"
)

OUT = (
    ROOT / "sam3_robowm/identity_v1/"
    "sam_masked_identity_pilot"
)
OUT.mkdir(parents=True, exist_ok=True)

CROP_OUT = OUT / "crops"
DEBUG_OUT = OUT / "debug"
SOURCE_OUT = OUT / "source"

CROP_OUT.mkdir(exist_ok=True)
DEBUG_OUT.mkdir(exist_ok=True)
SOURCE_OUT.mkdir(exist_ok=True)

BPE = Path(
    "/shared/ssd_30T/zhuoyingyang/physact/"
    "ReVidgen/pkgs/Grounded-Segment-Anything/"
    "playground/ImageBind_SAM/bpe/"
    "bpe_simple_vocab_16e6.txt.gz"
)

SAM_CONFIDENCE = 0.05

VIDEOS = {
    "Cosmos2.5_0016":
        ROOT / "cosmos_predict25/native_outputs/"
        "robowm_68_full/0016.mp4",

    "Cosmos3_seed101_0003":
        ROOT / "cosmos3/export_robowm68_3seeds/"
        "seed101/0003.mp4",

    "LVP_0004":
        ROOT / "evaluation_results/"
        "robowm_lvp_68_numbered/videos/0004.mp4",
}

# Diagnostic frames only.
# Includes true duplication windows + known false-positive periods.
FRAMES = {
    "Cosmos2.5_0016": [
        21, 27, 33, 39, 60, 78
    ],

    "Cosmos3_seed101_0003": [
        30, 54, 99, 123,
        141, 144, 156, 162, 168, 177
    ],

    "LVP_0004": [
        9, 12, 15, 18, 30, 39, 45
    ],
}

# Only top source-similarity candidates are needed.
TOP_K = 3


def norm_box(x1, y1, x2, y2, W, H):
    cx = ((x1 + x2) / 2.0) / W
    cy = ((y1 + y2) / 2.0) / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H

    return [
        float(cx),
        float(cy),
        float(bw),
        float(bh),
    ]


def normalize_masks(masks, H, W):
    if masks is None:
        return np.zeros(
            (0, H, W),
            dtype=bool
        )

    if torch.is_tensor(masks):
        masks = (
            masks.detach()
            .float()
            .cpu()
            .numpy()
        )

    masks = np.asarray(masks)

    if masks.ndim == 4:
        masks = masks[:, 0]

    if masks.ndim == 2:
        masks = masks[None]

    out = []

    for m in masks:
        if m.shape != (H, W):
            m = cv2.resize(
                m.astype(np.float32),
                (W, H),
                interpolation=cv2.INTER_NEAREST,
            )

        out.append(m > 0.5)

    if not out:
        return np.zeros(
            (0, H, W),
            dtype=bool
        )

    return np.stack(out, axis=0)


def normalize_scores(scores, n):
    if scores is None:
        return np.zeros(
            n,
            dtype=np.float32
        )

    if torch.is_tensor(scores):
        scores = (
            scores.detach()
            .float()
            .cpu()
            .numpy()
        )

    scores = np.asarray(scores).reshape(-1)

    if len(scores) < n:
        scores = np.pad(
            scores,
            (0, n-len(scores)),
            constant_values=np.nan
        )

    return scores[:n]


def masked_object_crop(rgb, mask):
    ys, xs = np.where(mask)

    if len(xs) < 10:
        return None

    x1 = xs.min()
    x2 = xs.max()
    y1 = ys.min()
    y2 = ys.max()

    bw = max(1, x2-x1)
    bh = max(1, y2-y1)

    pad = int(
        0.08 * max(bw, bh)
    )

    H, W = mask.shape

    x1 = max(0, x1-pad)
    x2 = min(W-1, x2+pad)
    y1 = max(0, y1-pad)
    y2 = min(H-1, y2+pad)

    # Neutral gray background.
    obj = np.full_like(
        rgb,
        127
    )

    obj[mask] = rgb[mask]

    crop = obj[
        y1:y2+1,
        x1:x2+1
    ]

    return Image.fromarray(crop)


def run_sam_on_box(processor, pil_image, box):
    W, H = pil_image.size

    box_n = norm_box(
        box[0], box[1],
        box[2], box[3],
        W, H
    )

    with torch.inference_mode():

        if torch.cuda.is_available():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                state = processor.set_image(
                    pil_image
                )

                output = (
                    processor
                    .add_geometric_prompt(
                        state=state,
                        box=box_n,
                        label=True,
                    )
                )

        else:
            state = processor.set_image(
                pil_image
            )

            output = (
                processor
                .add_geometric_prompt(
                    state=state,
                    box=box_n,
                    label=True,
                )
            )

    masks = normalize_masks(
        output.get("masks", None),
        H,
        W
    )

    scores = normalize_scores(
        output.get("scores", None),
        len(masks)
    )

    if len(masks) == 0:
        return None, np.nan

    if np.all(np.isnan(scores)):
        best = 0
    else:
        best = int(
            np.nanargmax(scores)
        )

    return (
        masks[best],
        float(scores[best])
        if not np.isnan(scores[best])
        else np.nan
    )


# ============================================================
# LOAD SAM3
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("="*90)
print("SAM-MASKED IDENTITY PILOT")
print("="*90)
print("device:", device)
print("BPE:", BPE)

model = build_sam3_image_model(
    bpe_path=str(BPE),
    device=device,
    eval_mode=True,
)

processor = Sam3Processor(
    model,
    device=device,
    confidence_threshold=SAM_CONFIDENCE,
)

print("SAM3 ready.")


# ============================================================
# READ CANDIDATES
# ============================================================

with open(IN_CSV, newline="") as f:
    candidates = list(
        csv.DictReader(f)
    )

rows_out = []


# ============================================================
# SOURCE MASKS
# ============================================================

for case in VIDEOS:

    src = (
        SOURCE_ROOT
        / f"{case}_SOURCE_TARGET.jpg"
    )

    image = Image.open(
        src
    ).convert("RGB")

    W, H = image.size

    # Source target crop was already produced by GDINO.
    # Full crop becomes SAM box prompt.
    mask, sam_score = run_sam_on_box(
        processor,
        image,
        [0, 0, W-1, H-1]
    )

    if mask is None:
        print("SOURCE MASK FAILED:", case)
        continue

    rgb = np.asarray(image)

    crop = masked_object_crop(
        rgb,
        mask
    )

    if crop is None:
        print("SOURCE CROP FAILED:", case)
        continue

    out_path = (
        SOURCE_OUT
        / f"{case}.png"
    )

    crop.save(out_path)

    print(
        f"SOURCE {case}: "
        f"SAM={sam_score:.3f}"
    )


# ============================================================
# GENERATED CANDIDATES
# ============================================================

for case, video_path in VIDEOS.items():

    print("\n" + "="*90)
    print(case)
    print("="*90)

    case_rows = [
        r for r in candidates
        if r["case"] == case
        and int(r["frame"])
        in FRAMES[case]
    ]

    cap = cv2.VideoCapture(
        str(video_path)
    )

    for frame_idx in FRAMES[case]:

        g = [
            r for r in case_rows
            if int(r["frame"]) == frame_idx
        ]

        g = sorted(
            g,
            key=lambda r:
                float(
                    r["source_similarity"]
                ),
            reverse=True
        )[:TOP_K]

        if not g:
            print(
                f"frame={frame_idx:03d}: "
                "NO CANDIDATES"
            )
            continue

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_idx
        )

        ok, bgr = cap.read()

        if not ok:
            continue

        rgb = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB
        )

        pil = Image.fromarray(rgb)

        frame_debug = bgr.copy()

        for rank, r in enumerate(
            g, 1
        ):

            box = [
                float(r["x1"]),
                float(r["y1"]),
                float(r["x2"]),
                float(r["y2"]),
            ]

            # Independent state for every box prompt.
            mask, sam_score = run_sam_on_box(
                processor,
                pil,
                box
            )

            if mask is None:
                print(
                    case,
                    frame_idx,
                    rank,
                    "NO MASK"
                )
                continue

            crop = masked_object_crop(
                rgb,
                mask
            )

            if crop is None:
                continue

            case_dir = (
                CROP_OUT / case
            )
            case_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            crop_path = (
                case_dir
                / (
                    f"frame_{frame_idx:04d}"
                    f"_rank_{rank:02d}.png"
                )
            )

            crop.save(crop_path)

            # mask / bbox diagnostics
            x1, y1, x2, y2 = [
                int(v)
                for v in box
            ]

            cv2.rectangle(
                frame_debug,
                (x1, y1),
                (x2, y2),
                (0,255,0),
                2
            )

            contours, _ = (
                cv2.findContours(
                    mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
            )

            cv2.drawContours(
                frame_debug,
                contours,
                -1,
                (0,255,255),
                2
            )

            rows_out.append({
                "case": case,
                "frame": frame_idx,
                "rank": rank,
                "old_source_similarity":
                    float(
                        r["source_similarity"]
                    ),
                "gdino_score":
                    float(r["gdino_score"]),
                "sam_score":
                    sam_score,
                "mask_pixels":
                    int(mask.sum()),
                "crop_path":
                    str(crop_path),
            })

        debug_dir = (
            DEBUG_OUT / case
        )
        debug_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        cv2.imwrite(
            str(
                debug_dir
                / f"frame_{frame_idx:04d}.jpg"
            ),
            frame_debug
        )

        print(
            f"frame={frame_idx:03d} "
            f"processed={len(g)}"
        )

    cap.release()


# ============================================================
# SAVE METADATA
# ============================================================

meta = (
    OUT
    / "SAM_MASKED_CANDIDATES.csv"
)

with open(
    meta,
    "w",
    newline=""
) as f:

    fields = [
        "case",
        "frame",
        "rank",
        "old_source_similarity",
        "gdino_score",
        "sam_score",
        "mask_pixels",
        "crop_path",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()
    writer.writerows(rows_out)


print("\n" + "="*90)
print("DONE")
print("="*90)
print("metadata:", meta)
print("sources :", SOURCE_OUT)
print("crops   :", CROP_OUT)
print("debug   :", DEBUG_OUT)

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


ROOT = Path("/shared/ssd_30T/zhuoyingyang/physact")
BASE = ROOT / "sam3_robowm/identity_v1"

MANIFEST = BASE / "/shared/ssd_30T/zhuoyingyang/physact/sam3_robowm/identity_v1/sam_identity_eval_casewise/manifests/Cosmos3_seed103_0011.csv"

SIGLIP_BASE = (
    BASE / "siglip_identity_eval_v1"
)

CANDIDATE_CSV = (
    SIGLIP_BASE / "SIGLIP_IDENTITY_EVAL_V1_CANDIDATES.csv"
)

SOURCE_IN = (
    SIGLIP_BASE / "source_targets"
)

OUT = (
    BASE / "/shared/ssd_30T/zhuoyingyang/physact/sam3_robowm/identity_v1/sam_identity_eval_casewise/case_Cosmos3_seed103_0011"
)

CROP_OUT = OUT / "crops"
SOURCE_OUT = OUT / "source"
DEBUG_OUT = OUT / "debug"

for p in [OUT, CROP_OUT, SOURCE_OUT, DEBUG_OUT]:
    p.mkdir(parents=True, exist_ok=True)


BPE = Path(
    "/shared/ssd_30T/zhuoyingyang/physact/"
    "ReVidgen/pkgs/Grounded-Segment-Anything/"
    "playground/ImageBind_SAM/bpe/"
    "bpe_simple_vocab_16e6.txt.gz"
)

SAM_CONFIDENCE = 0.05

# Frozen proposal selection from development:
# only top-3 SigLIP source-ranked proposals.
TOP_K = 3


def norm_box(x1, y1, x2, y2, W, H):
    return [
        ((x1+x2)/2) / W,
        ((y1+y2)/2) / H,
        (x2-x1) / W,
        (y2-y1) / H,
    ]


def normalize_masks(masks, H, W):
    if masks is None:
        return np.zeros(
            (0,H,W),
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
        masks = masks[:,0]

    if masks.ndim == 2:
        masks = masks[None]

    out = []

    for m in masks:
        if m.shape != (H,W):
            m = cv2.resize(
                m.astype(np.float32),
                (W,H),
                interpolation=cv2.INTER_NEAREST,
            )

        out.append(m > 0.5)

    if not out:
        return np.zeros(
            (0,H,W),
            dtype=bool
        )

    return np.stack(out)


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
            (0,n-len(scores)),
            constant_values=np.nan
        )

    return scores[:n]


def masked_crop(rgb, mask):
    ys, xs = np.where(mask)

    if len(xs) < 10:
        return None

    x1,x2 = xs.min(),xs.max()
    y1,y2 = ys.min(),ys.max()

    bw = max(1,x2-x1)
    bh = max(1,y2-y1)

    pad = int(
        0.08 * max(bw,bh)
    )

    H,W = mask.shape

    x1 = max(0,x1-pad)
    x2 = min(W-1,x2+pad)
    y1 = max(0,y1-pad)
    y2 = min(H-1,y2+pad)

    # Same neutral background as development pilot.
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


def run_sam(processor, pil, box):
    W,H = pil.size

    b = norm_box(
        box[0],box[1],
        box[2],box[3],
        W,H
    )

    with torch.inference_mode():

        if torch.cuda.is_available():

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):

                state = processor.set_image(
                    pil
                )

                output = (
                    processor.add_geometric_prompt(
                        state=state,
                        box=b,
                        label=True,
                    )
                )

        else:

            state = processor.set_image(
                pil
            )

            output = (
                processor.add_geometric_prompt(
                    state=state,
                    box=b,
                    label=True,
                )
            )

    masks = normalize_masks(
        output.get("masks",None),
        H,W
    )

    scores = normalize_scores(
        output.get("scores",None),
        len(masks)
    )

    if len(masks) == 0:
        return None,np.nan

    if np.all(np.isnan(scores)):
        best = 0
    else:
        best = int(
            np.nanargmax(scores)
        )

    score = (
        float(scores[best])
        if not np.isnan(scores[best])
        else np.nan
    )

    return masks[best],score


# ============================================================
# LOAD
# ============================================================

manifest = pd.read_csv(MANIFEST)
cand = pd.read_csv(CANDIDATE_CSV)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("="*100)
print("SAM3 MASKED EXPANSION-6")
print("="*100)
print("device:",device)

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
# SOURCE TARGETS
# ============================================================

for _,cfg in manifest.iterrows():

    case = cfg["case"]

    src_path = (
        SOURCE_IN
        / f"{case}_SOURCE_TARGET.jpg"
    )

    img = Image.open(
        src_path
    ).convert("RGB")

    W,H = img.size

    mask,score = run_sam(
        processor,
        img,
        [0,0,W-1,H-1]
    )

    if mask is None:
        print(
            "SOURCE MASK FAILED:",
            case
        )
        continue

    crop = masked_crop(
        np.asarray(img),
        mask
    )

    if crop is None:
        print(
            "SOURCE CROP FAILED:",
            case
        )
        continue

    crop.save(
        SOURCE_OUT / f"{case}.png"
    )

    print(
        f"SOURCE {case}: "
        f"SAM={score:.3f}"
    )


# ============================================================
# GENERATED VIDEO TOP-3
# ============================================================

rows = []

for _,cfg in manifest.iterrows():

    case = cfg["case"]
    video = Path(
        cfg["video_path"]
    )

    print("\n"+"="*100)
    print(case)
    print("="*100)

    z = cand[
        cand["case"] == case
    ].copy()

    # Frozen: retain only source-ranked top-3.
    z = z[
        z["rank"] <= TOP_K
    ].copy()

    cap = cv2.VideoCapture(
        str(video)
    )

    frames = sorted(
        z["frame"]
        .astype(int)
        .unique()
    )

    for frame in frames:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame)
        )

        ok,bgr = cap.read()

        if not ok:
            continue

        rgb = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB
        )

        pil = Image.fromarray(
            rgb
        )

        g = (
            z[z["frame"] == frame]
            .sort_values("rank")
        )

        n_ok = 0

        for _,r in g.iterrows():

            rank = int(r["rank"])

            box = [
                float(r["x1"]),
                float(r["y1"]),
                float(r["x2"]),
                float(r["y2"]),
            ]

            mask,sam_score = run_sam(
                processor,
                pil,
                box
            )

            if mask is None:
                continue

            crop = masked_crop(
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
                    f"frame_{int(frame):04d}"
                    f"_rank_{rank:02d}.png"
                )
            )

            crop.save(
                crop_path
            )

            rows.append({
                "case":case,
                "frame":int(frame),
                "rank":rank,

                "gdino_score":
                    float(r["gdino_score"]),

                "source_similarity":
                    float(
                        r["source_similarity"]
                    ),

                # Keep original proposal geometry
                # for containment calculation later.
                "x1":box[0],
                "y1":box[1],
                "x2":box[2],
                "y2":box[3],

                "sam_score":
                    sam_score,

                "mask_pixels":
                    int(mask.sum()),

                "crop_path":
                    str(crop_path),
            })

            n_ok += 1

        print(
            f"frame={int(frame):03d} "
            f"SAM crops={n_ok}"
        )

    cap.release()


out = pd.DataFrame(rows)

csv_path = (
    OUT
    / "SAM_CASE_CANDIDATES.csv"
)

out.to_csv(
    csv_path,
    index=False
)

print("\n"+"="*100)
print("DONE")
print("="*100)
print("CSV:",csv_path)
print("SOURCES:",SOURCE_OUT)
print("CROPS:",CROP_OUT)

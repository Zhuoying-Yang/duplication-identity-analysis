from pathlib import Path
import csv
import cv2
import numpy as np
import torch
from PIL import Image

from groundingdino.util.inference import load_model, load_image, predict
from transformers import AutoImageProcessor, SiglipModel


ROOT = Path("/shared/ssd_30T/zhuoyingyang/physact")

GDINO = (
    ROOT / "ReVidgen/pkgs/Grounded-Segment-Anything/GroundingDINO"
)
CONFIG = GDINO / "groundingdino/config/GroundingDINO_SwinB.py"
CHECKPOINT = (
    ROOT / "ReVidgen/checkpoints/GroundingDino/"
    "groundingdino_swinb_cogcoor.pth"
)

SIGLIP_CACHE = (
    ROOT / "cosmos_predict25/hf_cache/hub/"
    "models--google--siglip-so400m-patch14-384"
)

snapshots = sorted((SIGLIP_CACHE / "snapshots").glob("*"))
if not snapshots:
    raise RuntimeError(f"No SigLIP snapshot found: {SIGLIP_CACHE}")

SIGLIP_PATH = snapshots[-1]

SOURCE_DIR = (
    ROOT / "evaluation_results/"
    "robowm_lvp_68_numbered/images"
)

OUT = (
    ROOT / "sam3_robowm/identity_v1/"
    "siglip_source_match_pilot"
)
OUT.mkdir(parents=True, exist_ok=True)

TMP = OUT / "_tmp"
TMP.mkdir(exist_ok=True)

DEBUG = OUT / "debug"
DEBUG.mkdir(exist_ok=True)


CASES = {
    "Cosmos2.5_0016": {
        "video": (
            ROOT / "cosmos_predict25/native_outputs/"
            "robowm_68_full/0016.mp4"
        ),
        "idx": 16,
        "prompt": "white cup",
    },

    "Cosmos3_seed101_0003": {
        "video": (
            ROOT / "cosmos3/export_robowm68_3seeds/"
            "seed101/0003.mp4"
        ),
        "idx": 3,
        "prompt": "yellow cube",
    },

    "LVP_0004": {
        "video": (
            ROOT / "evaluation_results/"
            "robowm_lvp_68_numbered/videos/0004.mp4"
        ),
        "idx": 4,
        "prompt": "yellow cube",
    },
}


FRAME_STEP = 3

GDINO_BOX_THRESHOLD = 0.10
TEXT_THRESHOLD = 0.10

NMS_IOU = 0.50
MAX_PROPOSALS = 12

# Give crop a little context.
CROP_EXPAND = 0.08


def find_source(idx):
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        p = SOURCE_DIR / f"{idx:04d}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"source not found: {idx:04d}")


def cxcywh_to_xyxy(boxes, W, H):
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    b = np.asarray(boxes, dtype=np.float32)

    cx = b[:, 0] * W
    cy = b[:, 1] * H
    bw = b[:, 2] * W
    bh = b[:, 3] * H

    return np.stack([
        cx - bw/2,
        cy - bh/2,
        cx + bw/2,
        cy + bh/2
    ], axis=1)


def iou(box, boxes):
    if len(boxes) == 0:
        return np.array([])

    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    inter = (
        np.maximum(0, xx2-xx1)
        * np.maximum(0, yy2-yy1)
    )

    a = max(0, box[2]-box[0]) * max(0, box[3]-box[1])

    b = (
        np.maximum(0, boxes[:, 2]-boxes[:, 0])
        * np.maximum(0, boxes[:, 3]-boxes[:, 1])
    )

    return inter / (a + b - inter + 1e-6)


def nms(boxes, scores):
    if len(boxes) == 0:
        return []

    order = np.argsort(scores)[::-1]
    keep = []

    while len(order):
        i = int(order[0])
        keep.append(i)

        if len(order) == 1:
            break

        rest = order[1:]
        ov = iou(boxes[i], boxes[rest])

        order = rest[ov <= NMS_IOU]

    return keep


def crop_box(image_rgb, box):
    H, W = image_rgb.shape[:2]

    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1

    x1 -= CROP_EXPAND * bw
    x2 += CROP_EXPAND * bw
    y1 -= CROP_EXPAND * bh
    y2 += CROP_EXPAND * bh

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(W, int(x2))
    y2 = min(H, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return Image.fromarray(
        image_rgb[y1:y2, x1:x2]
    )


device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 90)
print("SOURCE-CONDITIONED IDENTITY PILOT")
print("=" * 90)
print("device:", device)
print("SigLIP:", SIGLIP_PATH)


print("\nLoading GroundingDINO...")

gdino = load_model(
    str(CONFIG),
    str(CHECKPOINT),
    device=device
)

print("GroundingDINO ready.")


print("\nLoading SigLIP...")

processor = AutoImageProcessor.from_pretrained(
    str(SIGLIP_PATH),
    local_files_only=True,
)

siglip = SiglipModel.from_pretrained(
    str(SIGLIP_PATH),
    local_files_only=True,
).to(device).eval()

print("SigLIP ready.")


@torch.no_grad()
def embed_images(images):
    inp = processor(
        images=images,
        return_tensors="pt"
    )

    pixel_values = inp["pixel_values"].to(device)

    feats = siglip.get_image_features(
        pixel_values=pixel_values
    )

    feats = feats / (
        feats.norm(dim=-1, keepdim=True) + 1e-8
    )

    return feats


candidate_rows = []
summary_rows = []


for case, cfg in CASES.items():

    print("\n" + "="*90)
    print(case, "|", cfg["prompt"])
    print("="*90)

    source_path = find_source(cfg["idx"])

    # ----------------------------------------------------
    # Find SOURCE target using GroundingDINO.
    # Use highest-confidence source proposal.
    # ----------------------------------------------------

    src_bgr = cv2.imread(str(source_path))
    src_rgb = cv2.cvtColor(
        src_bgr,
        cv2.COLOR_BGR2RGB
    )

    Hs, Ws = src_rgb.shape[:2]

    src_tmp = TMP / f"{case}_source.jpg"
    cv2.imwrite(str(src_tmp), src_bgr)

    _, src_tensor = load_image(str(src_tmp))

    boxes, logits, phrases = predict(
        model=gdino,
        image=src_tensor,
        caption=cfg["prompt"],
        box_threshold=GDINO_BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=device
    )

    boxes = boxes.detach().cpu().numpy()
    logits = logits.detach().cpu().numpy().reshape(-1)

    src_boxes = cxcywh_to_xyxy(
        boxes,
        Ws,
        Hs
    )

    keep = nms(src_boxes, logits)

    if not keep:
        raise RuntimeError(
            f"No source detection for {case}"
        )

    best = keep[0]

    source_crop = crop_box(
        src_rgb,
        src_boxes[best]
    )

    source_crop.save(
        OUT / f"{case}_SOURCE_TARGET.jpg"
    )

    source_feat = embed_images(
        [source_crop]
    )[0]

    print(
        f"SOURCE bbox score={logits[best]:.3f}"
    )

    # ----------------------------------------------------
    # Generated video
    # ----------------------------------------------------

    cap = cv2.VideoCapture(
        str(cfg["video"])
    )

    total = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    frame_ids = list(
        range(0, total, FRAME_STEP)
    )

    if total - 1 not in frame_ids:
        frame_ids.append(total - 1)

    case_debug = DEBUG / case
    case_debug.mkdir(exist_ok=True)

    for frame_idx in frame_ids:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_idx
        )

        ok, frame_bgr = cap.read()

        if not ok:
            continue

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB
        )

        H, W = frame_rgb.shape[:2]

        tmp = TMP / f"{case}_{frame_idx:04d}.jpg"
        cv2.imwrite(str(tmp), frame_bgr)

        _, image_tensor = load_image(str(tmp))

        boxes, logits, phrases = predict(
            model=gdino,
            image=image_tensor,
            caption=cfg["prompt"],
            box_threshold=GDINO_BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=device
        )

        boxes = boxes.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy().reshape(-1)

        xyxy = cxcywh_to_xyxy(
            boxes,
            W,
            H
        )

        keep = nms(
            xyxy,
            logits
        )

        keep = keep[:MAX_PROPOSALS]

        crops = []
        valid = []

        for i in keep:
            crop = crop_box(
                frame_rgb,
                xyxy[i]
            )

            if crop is None:
                continue

            crops.append(crop)
            valid.append(i)

        sims = []

        if crops:
            feats = embed_images(crops)

            sims = (
                feats @ source_feat
            ).detach().cpu().numpy()

        frame_candidates = []

        for rank, (i, sim) in enumerate(
            zip(valid, sims),
            1
        ):
            x1, y1, x2, y2 = xyxy[i]

            row = {
                "case": case,
                "frame": frame_idx,
                "rank": rank,
                "gdino_score": float(logits[i]),
                "source_similarity": float(sim),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }

            candidate_rows.append(row)
            frame_candidates.append(row)

        # Rank by SOURCE similarity, not GDINO score
        frame_candidates.sort(
            key=lambda x:
                x["source_similarity"],
            reverse=True
        )

        sim1 = (
            frame_candidates[0]["source_similarity"]
            if len(frame_candidates) >= 1
            else np.nan
        )

        sim2 = (
            frame_candidates[1]["source_similarity"]
            if len(frame_candidates) >= 2
            else np.nan
        )

        summary_rows.append({
            "case": case,
            "frame": frame_idx,
            "n_proposals": len(frame_candidates),
            "top1_similarity": sim1,
            "top2_similarity": sim2,
        })

        print(
            f"frame={frame_idx:03d} "
            f"n={len(frame_candidates):2d} "
            f"sim1={sim1:6.3f} "
            f"sim2={sim2:6.3f}"
        )

        # ------------------------------------------------
        # Debug: top 4 by source similarity
        # ------------------------------------------------

        vis = frame_bgr.copy()

        for j, r in enumerate(
            frame_candidates[:4]
        ):
            x1 = int(r["x1"])
            y1 = int(r["y1"])
            x2 = int(r["x2"])
            y2 = int(r["y2"])

            cv2.rectangle(
                vis,
                (x1,y1),
                (x2,y2),
                (0,255,0),
                2
            )

            cv2.putText(
                vis,
                (
                    f"S={r['source_similarity']:.2f} "
                    f"G={r['gdino_score']:.2f}"
                ),
                (x1, max(20,y1-5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                2
            )

        cv2.imwrite(
            str(
                case_debug
                / f"frame_{frame_idx:04d}.jpg"
            ),
            vis
        )

    cap.release()


# ============================================================
# SAVE
# ============================================================

candidate_csv = OUT / "SIGLIP_CANDIDATES.csv"
summary_csv = OUT / "SIGLIP_FRAME_SUMMARY.csv"

with open(
    candidate_csv,
    "w",
    newline=""
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=candidate_rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(candidate_rows)

with open(
    summary_csv,
    "w",
    newline=""
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=summary_rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(summary_rows)


print("\n" + "="*90)
print("DONE")
print("="*90)
print(candidate_csv)
print(summary_csv)
print(DEBUG)

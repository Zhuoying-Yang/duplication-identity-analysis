from pathlib import Path
import csv
import cv2
import numpy as np
import torch

from groundingdino.util.inference import (
    load_model,
    load_image,
    predict,
)

# ============================================================
# PATHS
# ============================================================

ROOT = Path("/shared/ssd_30T/zhuoyingyang/physact")

GDINO = (
    ROOT
    / "ReVidgen"
    / "pkgs"
    / "Grounded-Segment-Anything"
    / "GroundingDINO"
)

CONFIG = (
    GDINO
    / "groundingdino"
    / "config"
    / "GroundingDINO_SwinB.py"
)

CHECKPOINT = (
    ROOT
    / "ReVidgen"
    / "checkpoints"
    / "GroundingDino"
    / "groundingdino_swinb_cogcoor.pth"
)

OUT = (
    ROOT
    / "sam3_robowm"
    / "identity_v1"
    / "gdino_negative6_frozen"
)

TMP = OUT / "_tmp"
DEBUG = OUT / "debug_frames"

OUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)
DEBUG.mkdir(parents=True, exist_ok=True)

# ============================================================
# CASES
# ============================================================

MANIFEST = (
    Path(__file__).resolve().parent
    / "NEGATIVE6_FROZEN.csv"
)

CASES = {}

with open(
    MANIFEST,
    newline=""
) as f:
    for r in csv.DictReader(f):
        CASES[r["case"]] = {
            "video": Path(r["video_path"]),
            "prompt": r["object_prompt"],
        }

# Frozen sampling
FRAME_STEP = 3

# Detection itself is permissive.
# Later we inspect counts at multiple confidence thresholds.
PREDICT_BOX_THRESHOLD = 0.10
TEXT_THRESHOLD = 0.10

COUNT_THRESHOLDS = [
    0.12,
    0.20,
    0.30,
]

# Same-object duplicate boxes are merged.
NMS_IOU = 0.50


# ============================================================
# HELPERS
# ============================================================

def cxcywh_to_xyxy(boxes, W, H):
    """
    GroundingDINO predict() returns normalized cx,cy,w,h.
    Convert to pixel x1,y1,x2,y2.
    """
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    b = np.asarray(boxes, dtype=np.float32)

    cx = b[:, 0] * W
    cy = b[:, 1] * H
    bw = b[:, 2] * W
    bh = b[:, 3] * H

    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    out = np.stack(
        [x1, y1, x2, y2],
        axis=1
    )

    out[:, [0, 2]] = np.clip(
        out[:, [0, 2]],
        0,
        W - 1
    )

    out[:, [1, 3]] = np.clip(
        out[:, [1, 3]],
        0,
        H - 1
    )

    return out


def iou_one_to_many(box, boxes):
    if len(boxes) == 0:
        return np.array([])

    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    inter = w * h

    area1 = max(
        0,
        box[2] - box[0]
    ) * max(
        0,
        box[3] - box[1]
    )

    area2 = np.maximum(
        0,
        boxes[:, 2] - boxes[:, 0]
    ) * np.maximum(
        0,
        boxes[:, 3] - boxes[:, 1]
    )

    union = area1 + area2 - inter + 1e-6

    return inter / union


def nms_numpy(boxes, scores, iou_thr=0.5):
    if len(boxes) == 0:
        return []

    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        i = int(order[0])
        keep.append(i)

        if len(order) == 1:
            break

        rest = order[1:]

        ious = iou_one_to_many(
            boxes[i],
            boxes[rest]
        )

        order = rest[
            ious <= iou_thr
        ]

    return keep


def get_video_frame(cap, idx):
    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        idx
    )

    ok, frame = cap.read()

    if not ok:
        return None

    return frame


# ============================================================
# LOAD MODEL
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 100)
print("GROUNDINGDINO IDENTITY COUNT PILOT")
print("=" * 100)
print("device:", device)
print("config:", CONFIG)
print("checkpoint:", CHECKPOINT)
print()

model = load_model(
    str(CONFIG),
    str(CHECKPOINT),
    device=device,
)

print("GroundingDINO ready.")



def proposal_iou_one_to_many(box, boxes):
    """
    box: [4] xyxy
    boxes: [N,4] xyxy
    """
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float32)

    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])

    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)

    inter = iw * ih

    area_a = max(
        1e-6,
        (box[2] - box[0]) *
        (box[3] - box[1])
    )

    area_b = np.maximum(
        1e-6,
        (boxes[:, 2] - boxes[:, 0]) *
        (boxes[:, 3] - boxes[:, 1])
    )

    union = area_a + area_b - inter

    return inter / np.maximum(union, 1e-6)


def nms(boxes, scores):
    if len(boxes) == 0:
        return []

    boxes = np.asarray(boxes)
    scores = np.asarray(scores)

    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        i = int(order[0])
        keep.append(i)

        if len(order) == 1:
            break

        rest = order[1:]

        overlaps = proposal_iou_one_to_many(
            boxes[i],
            boxes[rest]
        )

        order = rest[
            overlaps <= NMS_IOU
        ]

    return keep


# ============================================================
# RUN
# ============================================================

all_rows = []
proposal_rows = []

for case, cfg in CASES.items():

    video_path = cfg["video"]
    prompt = cfg["prompt"]

    cap = cv2.VideoCapture(
        str(video_path)
    )

    total = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print()
    print("=" * 100)
    print(case)
    print("prompt:", prompt)
    print("frames:", total)
    print("frozen negative diagnostic; no oracle timing used")
    print("=" * 100)

    case_debug = DEBUG / case
    case_debug.mkdir(
        parents=True,
        exist_ok=True
    )

    frame_ids = list(
        range(
            0,
            total,
            FRAME_STEP
        )
    )

    if total > 0 and total - 1 not in frame_ids:
        frame_ids.append(total - 1)

    for frame_idx in frame_ids:

        frame = get_video_frame(
            cap,
            frame_idx
        )

        if frame is None:
            continue

        H, W = frame.shape[:2]

        tmp_path = (
            TMP
            / f"{case}_{frame_idx:04d}.jpg"
        )

        cv2.imwrite(
            str(tmp_path),
            frame
        )

        image_source, image = load_image(
            str(tmp_path)
        )

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=prompt,
            box_threshold=PREDICT_BOX_THRESHOLD,
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
        logits = np.asarray(logits).reshape(-1)

        xyxy = cxcywh_to_xyxy(
            boxes,
            W,
            H
        )

        counts = {}
        keeps_by_thr = {}

        for thr in COUNT_THRESHOLDS:

            valid = np.where(
                logits >= thr
            )[0]

            if len(valid) == 0:
                counts[thr] = 0
                keeps_by_thr[thr] = []
                continue

            b = xyxy[valid]
            s = logits[valid]

            keep_local = nms_numpy(
                b,
                s,
                NMS_IOU
            )

            keep_global = [
                int(valid[k])
                for k in keep_local
            ]

            counts[thr] = len(
                keep_global
            )

            keeps_by_thr[thr] = (
                keep_global
            )

        row = {
            "case": case,
            "frame": frame_idx,
            "prompt": prompt,
            "raw_detections": len(logits),

            "count_012":
                counts[0.12],

            "count_020":
                counts[0.20],

            "count_030":
                counts[0.30],

            "max_score":
                float(logits.max())
                if len(logits)
                else 0.0,
        }

        # ------------------------------------------------
        # SAVE FROZEN NMS PROPOSALS @ threshold 0.12
        # ------------------------------------------------
        # Frozen SigLIP pilot used GDINO detection threshold 0.10,
        # followed by NMS and MAX_PROPOSALS=12.
        proposal_keep = nms(
            xyxy,
            logits
        )

        proposal_keep = sorted(
            proposal_keep,
            key=lambda i: float(logits[i]),
            reverse=True,
        )[:12]

        for rank, det_i in enumerate(
            proposal_keep,
            start=1
        ):
            x1, y1, x2, y2 = [
                float(v)
                for v in xyxy[det_i]
            ]

            phrase = (
                str(phrases[det_i])
                if det_i < len(phrases)
                else ""
            )

            proposal_rows.append({
                "case": case,
                "frame": frame_idx,
                "prompt": prompt,
                "rank": rank,
                "gdino_score":
                    float(logits[det_i]),
                "phrase": phrase,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            })

        all_rows.append(row)

        print(
            f"frame={frame_idx:03d} "
            f"raw={len(logits):2d} "
            f"NMS counts "
            f"0.12={counts[0.12]} "
            f"0.20={counts[0.20]} "
            f"0.30={counts[0.30]} "
            f"max={row['max_score']:.3f}"
        )

        # ------------------------------------------------
        # DEBUG IMAGE
        # Draw medium threshold = 0.20
        # ------------------------------------------------

        vis = frame.copy()

        keep = keeps_by_thr[
            0.20
        ]

        for det_i in keep:

            x1, y1, x2, y2 = (
                xyxy[det_i]
                .astype(int)
            )

            score = float(
                logits[det_i]
            )

            cv2.rectangle(
                vis,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                vis,
                f"{prompt} {score:.2f}",
                (x1, max(20, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            vis,
            (
                f"FRAME {frame_idx} | "
                f"count@0.20={counts[0.20]}"
            ),
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
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
# SAVE CSV
# ============================================================

csv_path = (
    OUT
    / "GDINO_NEGATIVE6_COUNTS.csv"
)

with open(
    csv_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "case",
            "frame",
            "prompt",
            "raw_detections",
            "count_012",
            "count_020",
            "count_030",
            "max_score",
        ]
    )

    writer.writeheader()
    writer.writerows(all_rows)

proposal_csv = (
    OUT
    / "GDINO_NEGATIVE6_PROPOSALS.csv"
)

with open(
    proposal_csv,
    "w",
    newline=""
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "case",
            "frame",
            "prompt",
            "rank",
            "gdino_score",
            "phrase",
            "x1",
            "y1",
            "x2",
            "y2",
        ]
    )

    writer.writeheader()
    writer.writerows(
        proposal_rows
    )

print()
print("=" * 100)
print("DONE")
print("=" * 100)
print("COUNTS:", csv_path)
print("PROPOSALS:", proposal_csv)
print("debug:", DEBUG)

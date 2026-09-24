import cv2
import itertools
import tempfile
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageDraw

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/shared/ssd_30T/zhuoyingyang/physact"
)

MODEL_PATH = (
    "/shared/ssd_30T/zhuoyingyang/models/"
    "Qwen3-VL-8B-Instruct"
)

BASE = (
    ROOT
    / "sam3_robowm/identity_v1"
)

SAM_BASE = (
    BASE
    / "sam_masked_identity_pilot"
)

META_CSV = (
    SAM_BASE
    / "SAM_MASKED_CANDIDATES.csv"
)

GDINO_CSV = (
    BASE
    / "siglip_source_match_pilot/"
    "SIGLIP_CANDIDATES.csv"
)

SOURCE_DIR = (
    SAM_BASE
    / "source"
)

OUT_PAIR = (
    BASE
    / "QWEN_SAM_PAIR_PILOT_PAIRS.csv"
)

OUT_FRAME = (
    BASE
    / "QWEN_SAM_PAIR_PILOT_FRAMES.csv"
)


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


# ============================================================
# DEVELOPMENT STRESS TEST
#
# These labels are ONLY for diagnostic interpretation.
# NOT a final evaluation set.
# ============================================================

FRAME_GT = {
    # Easy positive controls
    ("Cosmos2.5_0016", 33): "TRUE_DUP",
    ("Cosmos2.5_0016", 60): "TRUE_DUP",

    # True duplication interval
    ("Cosmos3_seed101_0003", 144): "TRUE_DUP",
    ("Cosmos3_seed101_0003", 156): "TRUE_DUP",
    ("Cosmos3_seed101_0003", 162): "TRUE_DUP",
    ("Cosmos3_seed101_0003", 168): "TRUE_DUP",

    # Hard false proposals in same positive video
    ("Cosmos3_seed101_0003", 30):  "NO_DUP",
    ("Cosmos3_seed101_0003", 54):  "NO_DUP",
    ("Cosmos3_seed101_0003", 99):  "NO_DUP",
    ("Cosmos3_seed101_0003", 123): "NO_DUP",

    # LVP disappearance case: no true duplication
    ("LVP_0004", 12): "NO_DUP",
    ("LVP_0004", 15): "NO_DUP",
    ("LVP_0004", 18): "NO_DUP",
    ("LVP_0004", 30): "NO_DUP",
}


PROMPT = """
You are verifying a proposed TARGET-OBJECT DUPLICATION event in ONE video frame.

INPUT ORDER:

1. REFERENCE TARGET:
   The task-relevant target object from the original source scene.

2. FULL FRAME CONTEXT:
   The generated video frame.
   Two proposed regions are marked:
   A = red box
   B = blue box.

3. CANDIDATE A:
   SAM-masked appearance of region A.

4. CANDIDATE B:
   SAM-masked appearance of region B.

Your ONLY question is:

"Do candidate A and candidate B represent TWO DISTINCT PHYSICAL INSTANCES
of the target object represented by REFERENCE, simultaneously present
in this frame?"

CLEAR YES requires BOTH:

- Candidate A is genuinely the target object or a plausible duplicated/
  mutated version of that target.
- Candidate B is also genuinely the target object or a plausible duplicated/
  mutated version of that target.
- They correspond to TWO spatially distinct physical object instances.

Answer NO if:

- one candidate is robot arm, gripper, table, drawer, trash can,
  background, or another unrelated object;
- A and B are two overlapping detections or two parts/crops of ONE object;
- a segmentation/detection artifact creates a false second candidate;
- only ONE real target instance exists;
- visual similarity is caused only by color or scene context.

Do NOT infer duplication merely because the two crops have similar color.

Choose exactly ONE:

A = NO, these are not two distinct target-object instances
B = UNCERTAIN / visually ambiguous
C = YES, these are clearly two distinct target-object instances

Answer exactly one letter: A, B, or C.
"""


# ============================================================
# HELPERS
# ============================================================

def token_id(processor, letter):
    for x in [letter, " " + letter]:
        ids = processor.tokenizer.encode(
            x,
            add_special_tokens=False,
        )
        if len(ids) == 1:
            return ids[0]

    raise RuntimeError(
        f"Cannot find single-token ID for {letter}"
    )


def label_image(img, text, width=640):
    img = img.convert("RGB")

    if img.width > width:
        scale = width / img.width
        img = img.resize(
            (
                width,
                int(img.height * scale),
            ),
            Image.Resampling.LANCZOS,
        )

    canvas = Image.new(
        "RGB",
        (
            img.width,
            img.height + 42,
        ),
        "black",
    )

    canvas.paste(
        img,
        (0, 42),
    )

    d = ImageDraw.Draw(canvas)

    d.text(
        (8, 12),
        text,
        fill="white",
    )

    return canvas


def read_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(
        str(video_path)
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(frame_idx),
    )

    ok, bgr = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(
            f"Cannot read {video_path} frame {frame_idx}"
        )

    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    )

    return Image.fromarray(rgb)


def draw_pair_context(
    video_path,
    frame_idx,
    box_a,
    box_b,
):
    img = read_frame(
        video_path,
        frame_idx,
    )

    d = ImageDraw.Draw(img)

    # A = RED
    d.rectangle(
        [
            box_a["x1"],
            box_a["y1"],
            box_a["x2"],
            box_a["y2"],
        ],
        outline="red",
        width=5,
    )

    d.text(
        (
            box_a["x1"] + 5,
            box_a["y1"] + 5,
        ),
        "A",
        fill="red",
        stroke_width=2,
        stroke_fill="black",
    )

    # B = BLUE
    d.rectangle(
        [
            box_b["x1"],
            box_b["y1"],
            box_b["x2"],
            box_b["y2"],
        ],
        outline="blue",
        width=5,
    )

    d.text(
        (
            box_b["x1"] + 5,
            box_b["y1"] + 5,
        ),
        "B",
        fill="blue",
        stroke_width=2,
        stroke_fill="white",
    )

    return img


# ============================================================
# LOAD DATA
# ============================================================

assert META_CSV.exists(), META_CSV
assert GDINO_CSV.exists(), GDINO_CSV

meta = pd.read_csv(META_CSV)
gdino = pd.read_csv(GDINO_CSV)

print("SAM metadata rows:", len(meta))
print("GDINO rows:", len(gdino))


# Recreate Stage-A rank:
# candidates were sorted by old source similarity descending
# and only TOP_K=3 were used.
box_lookup = {}

for (case, frame), g in gdino.groupby(
    ["case", "frame"]
):
    g = (
        g.sort_values(
            "source_similarity",
            ascending=False,
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


# ============================================================
# LOAD QWEN
# ============================================================

print("\nLoading Qwen3-VL...")

processor = AutoProcessor.from_pretrained(
    MODEL_PATH
)

model = (
    Qwen3VLForConditionalGeneration
    .from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
)

model.eval()

IDA = token_id(
    processor,
    "A",
)

IDB = token_id(
    processor,
    "B",
)

IDC = token_id(
    processor,
    "C",
)

print("Qwen ready.")
print(
    "token IDs:",
    IDA,
    IDB,
    IDC,
)


# ============================================================
# QWEN EVALUATION
# ============================================================

def evaluate_pair(
    source_path,
    context_img,
    crop_a_path,
    crop_b_path,
    case,
    frame,
    rank_a,
    rank_b,
):

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # 1. Reference target
        source = (
            Image.open(source_path)
            .convert("RGB")
        )

        source = label_image(
            source,
            "1. REFERENCE TARGET",
        )

        p_source = (
            td / "01_reference.jpg"
        )

        source.save(
            p_source,
            quality=95,
        )

        # 2. Full-frame context
        context = label_image(
            context_img,
            (
                f"2. FULL FRAME | {case} "
                f"frame {frame} | "
                f"A=rank{rank_a}, B=rank{rank_b}"
            ),
        )

        p_context = (
            td / "02_context.jpg"
        )

        context.save(
            p_context,
            quality=95,
        )

        # 3. Candidate A
        a = (
            Image.open(crop_a_path)
            .convert("RGB")
        )

        a = label_image(
            a,
            "3. CANDIDATE A - SAM masked",
        )

        p_a = (
            td / "03_candidate_A.jpg"
        )

        a.save(
            p_a,
            quality=95,
        )

        # 4. Candidate B
        b = (
            Image.open(crop_b_path)
            .convert("RGB")
        )

        b = label_image(
            b,
            "4. CANDIDATE B - SAM masked",
        )

        p_b = (
            td / "04_candidate_B.jpg"
        )

        b.save(
            p_b,
            quality=95,
        )

        files = [
            p_source,
            p_context,
            p_a,
            p_b,
        ]

        content = []

        for p in files:
            content.append({
                "type": "image",
                "image":
                    "file://"
                    + str(p.resolve()),
            })

        content.append({
            "type": "text",
            "text": PROMPT,
        })

        messages = [{
            "role": "user",
            "content": content,
        }]

        text = (
            processor
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        image_inputs, video_inputs = (
            process_vision_info(
                messages
            )
        )

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            out = model(
                **inputs
            )

        logits = (
            out.logits[0, -1]
            .float()
        )

        abc = torch.stack([
            logits[IDA],
            logits[IDB],
            logits[IDC],
        ])

        probs = (
            torch.softmax(
                abc,
                dim=0,
            )
            .cpu()
            .numpy()
        )

        pA, pB, pC = map(
            float,
            probs,
        )

        score = (
            100
            * (
                pC
                + 0.5 * pB
            )
        )

        pred = (
            ["A", "B", "C"][
                int(
                    probs.argmax()
                )
            ]
        )

        return (
            pred,
            score,
            pA,
            pB,
            pC,
        )


# ============================================================
# RUN ALL CANDIDATE PAIRS
# ============================================================

pair_rows = []

for (
    case,
    frame
), gt in FRAME_GT.items():

    print("\n" + "=" * 90)
    print(
        case,
        "| frame",
        frame,
        "| GT:",
        gt,
    )
    print("=" * 90)

    source_path = (
        SOURCE_DIR
        / f"{case}.png"
    )

    if not source_path.exists():
        print(
            "MISSING SOURCE:",
            source_path,
        )
        continue

    g = meta[
        (meta.case == case)
        & (meta.frame == frame)
    ].copy()

    g = (
        g.sort_values("rank")
        .reset_index(drop=True)
    )

    if len(g) < 2:
        print(
            "SKIP: fewer than 2 candidates"
        )
        continue

    # Test ALL available pairs among top-3 proposals.
    for i, j in itertools.combinations(
        range(len(g)),
        2,
    ):

        ra = g.iloc[i]
        rb = g.iloc[j]

        rank_a = int(ra["rank"])
        rank_b = int(rb["rank"])

        key_a = (
            case,
            frame,
            rank_a,
        )

        key_b = (
            case,
            frame,
            rank_b,
        )

        if (
            key_a not in box_lookup
            or key_b not in box_lookup
        ):
            print(
                "MISSING BOX:",
                key_a,
                key_b,
            )
            continue

        context = draw_pair_context(
            VIDEOS[case],
            frame,
            box_lookup[key_a],
            box_lookup[key_b],
        )

        result = evaluate_pair(
            source_path,
            context,
            Path(
                ra["crop_path"]
            ),
            Path(
                rb["crop_path"]
            ),
            case,
            frame,
            rank_a,
            rank_b,
        )

        pred, score, pA, pB, pC = result

        print(
            f"pair {rank_a}-{rank_b}: "
            f"pred={pred} "
            f"score={score:.2f} "
            f"A={pA:.3f} "
            f"B={pB:.3f} "
            f"C={pC:.3f}"
        )

        pair_rows.append({
            "case": case,
            "frame": frame,
            "gt": gt,
            "rank_a": rank_a,
            "rank_b": rank_b,

            "old_source_sim_a":
                float(
                    ra[
                        "old_source_similarity"
                    ]
                ),

            "old_source_sim_b":
                float(
                    rb[
                        "old_source_similarity"
                    ]
                ),

            "qwen_pred": pred,
            "qwen_score": score,
            "pA": pA,
            "pB": pB,
            "pC": pC,

            "crop_a":
                str(
                    ra["crop_path"]
                ),

            "crop_b":
                str(
                    rb["crop_path"]
                ),
        })


# ============================================================
# FRAME-LEVEL SCORE:
# max verified pair in that frame
# ============================================================

pairs = pd.DataFrame(
    pair_rows
)

pairs.to_csv(
    OUT_PAIR,
    index=False,
)

frame_rows = []

if len(pairs):

    for (
        case,
        frame,
        gt
    ), g in pairs.groupby(
        [
            "case",
            "frame",
            "gt",
        ]
    ):

        best = (
            g.sort_values(
                "qwen_score",
                ascending=False,
            )
            .iloc[0]
        )

        frame_rows.append({
            "case": case,
            "frame": frame,
            "gt": gt,

            "qwen_frame_score":
                float(
                    best.qwen_score
                ),

            "best_pair":
                (
                    f"{int(best.rank_a)}"
                    f"-"
                    f"{int(best.rank_b)}"
                ),

            "best_pred":
                best.qwen_pred,

            "best_pA":
                best.pA,

            "best_pB":
                best.pB,

            "best_pC":
                best.pC,
        })


frames = pd.DataFrame(
    frame_rows
)

frames.to_csv(
    OUT_FRAME,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("FRAME-LEVEL SUMMARY")
print("=" * 90)

if len(frames):

    print(
        frames.sort_values(
            [
                "gt",
                "qwen_frame_score",
            ],
            ascending=[
                True,
                False,
            ],
        ).to_string(
            index=False
        )
    )

    print("\n" + "-" * 90)

    for gt, g in frames.groupby(
        "gt"
    ):
        print(
            gt,
            "| N =",
            len(g),
            "| mean =",
            round(
                g.qwen_frame_score.mean(),
                2,
            ),
            "| median =",
            round(
                g.qwen_frame_score.median(),
                2,
            ),
            "| min =",
            round(
                g.qwen_frame_score.min(),
                2,
            ),
            "| max =",
            round(
                g.qwen_frame_score.max(),
                2,
            ),
        )


print("\nSaved:")
print(OUT_PAIR)
print(OUT_FRAME)

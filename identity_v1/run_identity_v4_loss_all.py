import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_PATH = "/shared/ssd_30T/zhuoyingyang/models/Qwen3-VL-8B-Instruct"

STRIDE = 4

# Relative to center t
BEFORE_OFFSETS = [-6, -4, -2]
AFTER_OFFSETS  = [ 2,  4,  6]

RESIZE_W = 640


PROMPT = """
You are evaluating OBJECT IDENTITY PERSISTENCE in a generated
robot manipulation video.

INPUT ORDER:

1. SOURCE IMAGE:
   the physical reference scene before video generation.

2. BEFORE frames:
   generated video frames immediately before a possible identity event.

3. AFTER frames:
   generated video frames immediately after the possible identity event.

Focus ONLY on the task-relevant target object.

The question is:

Does the SAME physical target identity that exists in the SOURCE / BEFORE
state continue coherently into the AFTER state?

A CLEAR IDENTITY-PERSISTENCE VIOLATION occurs if:

1. UNEXPLAINED DISAPPEARANCE:
   the target clearly exists BEFORE, but becomes absent AFTER without a
   physically plausible reason.

OR

2. IDENTITY REPLACEMENT / SWITCH:
   the original target identity is effectively abandoned or replaced by a
   newly appearing target-like object, and the manipulation continues using
   that new physical instance rather than the original target.

Important:

- Temporary occlusion by the robot/gripper is NOT a violation.
- Putting the target inside a container is NOT disappearance.
- Naturally moving the target outside the camera view is NOT disappearance.
- Normal appearance change of the same physical object is NOT identity loss.
- Do not infer events outside the shown BEFORE/AFTER frames.
- A duplicate may look identical or only approximately similar to the target.

Choose exactly ONE:

A = CONSISTENT
    The original target identity persists coherently.

B = AMBIGUOUS
    Identity persistence may be violated, but visual evidence is insufficient.

C = CLEAR IDENTITY-PERSISTENCE VIOLATION
    Clear unexplained disappearance or identity replacement/switch.

Answer with exactly one letter: A, B, or C.
"""


def token_id(processor, letter):
    for x in [letter, " " + letter]:
        ids = processor.tokenizer.encode(
            x,
            add_special_tokens=False
        )
        if len(ids) == 1:
            return ids[0]

    raise RuntimeError(letter)


def load_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))

    ok, frame = cap.read()

    if not ok:
        return None

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    return Image.fromarray(frame)


def label_image(img, text):

    img = img.convert("RGB")

    if img.width > RESIZE_W:
        scale = RESIZE_W / img.width

        img = img.resize(
            (
                RESIZE_W,
                int(img.height * scale)
            ),
            Image.Resampling.LANCZOS
        )

    canvas = Image.new(
        "RGB",
        (img.width, img.height + 38),
        "black"
    )

    canvas.paste(img, (0, 38))

    d = ImageDraw.Draw(canvas)

    d.text(
        (8, 10),
        text,
        fill="white"
    )

    return canvas


print("Loading processor...")
processor = AutoProcessor.from_pretrained(
    MODEL_PATH
)

print("Loading Qwen3-VL...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model.eval()

IDA = token_id(processor, "A")
IDB = token_id(processor, "B")
IDC = token_id(processor, "C")

print(
    "A/B/C token ids:",
    IDA,
    IDB,
    IDC
)


def score_pair(
    source_path,
    video_path,
    before_ids,
    after_ids,
    task_prompt
):

    with tempfile.TemporaryDirectory() as td:

        td = Path(td)

        files = []

        # ------------------------------------------------
        # SOURCE
        # ------------------------------------------------

        src = Image.open(
            source_path
        ).convert("RGB")

        src = label_image(
            src,
            "SOURCE IMAGE - physical reference"
        )

        p = td / "00_source.jpg"

        src.save(
            p,
            quality=95
        )

        files.append(p)

        # ------------------------------------------------
        # VIDEO
        # ------------------------------------------------

        cap = cv2.VideoCapture(
            video_path
        )

        for k, idx in enumerate(before_ids):

            img = load_frame(
                cap,
                idx
            )

            if img is None:
                continue

            img = label_image(
                img,
                f"BEFORE - frame {idx}"
            )

            p = td / f"before_{k}_{idx}.jpg"

            img.save(
                p,
                quality=92
            )

            files.append(p)

        for k, idx in enumerate(after_ids):

            img = load_frame(
                cap,
                idx
            )

            if img is None:
                continue

            img = label_image(
                img,
                f"AFTER - frame {idx}"
            )

            p = td / f"after_{k}_{idx}.jpg"

            img.save(
                p,
                quality=92
            )

            files.append(p)

        cap.release()

        full_prompt = f"""
TASK:
{task_prompt}

{PROMPT}
"""

        content = []

        for p in files:

            content.append({
                "type": "image",
                "image":
                    "file://" +
                    str(p.resolve())
            })

        content.append({
            "type": "text",
            "text": full_prompt
        })

        messages = [{
            "role": "user",
            "content": content
        }]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = (
            process_vision_info(messages)
        )

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        inputs = inputs.to(
            model.device
        )

        with torch.inference_mode():

            out = model(
                **inputs
            )

        logits = out.logits[
            0,
            -1
        ].float()

        abc = torch.stack([
            logits[IDA],
            logits[IDB],
            logits[IDC]
        ])

        probs = torch.softmax(
            abc,
            dim=0
        ).cpu().numpy()

        pA, pB, pC = [
            float(x)
            for x in probs
        ]

        score = 100.0 * (
            pC +
            0.5 * pB
        )

        pred = [
            "A",
            "B",
            "C"
        ][
            int(
                np.argmax(
                    probs
                )
            )
        ]

        return {
            "pred":
                pred,

            "pA":
                pA,

            "pB":
                pB,

            "pC_loss":
                pC,

            "loss_score":
                score
        }


df = pd.read_csv(
    "positives_v2_manifest.csv"
)

all_rows = []
summary = []


for _, r in df.iterrows():

    case = r["case"]

    video_path = r[
        "video_path"
    ]

    source_path = r[
        "source_path"
    ]

    task_prompt = r[
        "task_prompt"
    ]

    cap = cv2.VideoCapture(
        video_path
    )

    total = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()

    print()
    print("=" * 100)
    print(case)
    print(
        "frames:",
        total
    )
    print("=" * 100)

    # Need 6 frames on both sides
    centers = list(
        range(
            6,
            total - 6,
            STRIDE
        )
    )

    case_rows = []

    for j, t in enumerate(centers):

        before = [
            t + x
            for x in BEFORE_OFFSETS
        ]

        after = [
            t + x
            for x in AFTER_OFFSETS
        ]

        try:

            result = score_pair(
                source_path,
                video_path,
                before,
                after,
                task_prompt
            )

            row = {
                "case":
                    case,

                "center":
                    t,

                "before":
                    ",".join(
                        map(
                            str,
                            before
                        )
                    ),

                "after":
                    ",".join(
                        map(
                            str,
                            after
                        )
                    ),

                **result
            }

            case_rows.append(
                row
            )

            all_rows.append(
                row
            )

            print(
                f"[{j+1:02d}/{len(centers):02d}] "
                f"t={t:03d} "
                f"before={before} "
                f"after={after} "
                f"pred={result['pred']} "
                f"LOSS={result['loss_score']:.1f}"
            )

        except Exception as e:

            print(
                "ERROR",
                t,
                repr(e)
            )

    cdf = pd.DataFrame(
        case_rows
    )

    if len(cdf) == 0:
        continue

    best = cdf.loc[
        cdf[
            "loss_score"
        ].idxmax()
    ]

    summary.append({
        "case":
            case,

        "max_loss_score":
            float(
                best[
                    "loss_score"
                ]
            ),

        "max_loss_pred":
            best["pred"],

        "max_loss_center":
            int(
                best["center"]
            ),

        "max_loss_before":
            best["before"],

        "max_loss_after":
            best["after"],
    })

    pd.DataFrame(
        all_rows
    ).to_csv(
        "IDENTITY_V4_LOSS_ALL_WINDOWS.csv",
        index=False
    )

    pd.DataFrame(
        summary
    ).to_csv(
        "IDENTITY_V4_LOSS_SUMMARY.csv",
        index=False
    )


print()
print("=" * 100)
print("FINAL LOSS SUMMARY")
print("=" * 100)

s = pd.DataFrame(
    summary
)

print(
    s.to_string(
        index=False
    )
)

s.to_csv(
    "IDENTITY_V4_LOSS_SUMMARY.csv",
    index=False
)

pd.DataFrame(
    all_rows
).to_csv(
    "IDENTITY_V4_LOSS_ALL_WINDOWS.csv",
    index=False
)

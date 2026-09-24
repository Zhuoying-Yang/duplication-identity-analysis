import cv2
import pandas as pd
import torch
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_PATH = "/shared/ssd_30T/zhuoyingyang/models/Qwen3-VL-8B-Instruct"

# Manually verified GT windows.
# ONLY FOR ORACLE DIAGNOSTIC, NOT FINAL METRIC.
CASES = {
    "Cosmos2.5_0016": {
        "before": [24, 27, 30],
        "after":  [36, 40, 44],
        "gt": "DUPLICATION",
    },

    "Cosmos3_seed101_0003": {
        "before": [136, 139, 142],
        "after":  [150, 155, 160],
        "gt": "DUPLICATION",
    },

    "LVP_0004": {
        "before": [30, 32, 34],
        "after":  [38, 40, 42],
        "gt": "DISAPPEAR",
    },
}

DUP_PROMPT = """
You are evaluating TARGET OBJECT DUPLICATION in a generated robot manipulation video.

INPUT ORDER:
1. SOURCE IMAGE: physical reference scene before generation.
2. BEFORE frames: generated video frames immediately before the suspected event.
3. AFTER frames: generated video frames immediately after the suspected event.

Focus ONLY on the task-relevant target object.

A DUPLICATION violation means:

- The source/before state contains one physical target identity.
- In the AFTER state, an ADDITIONAL physical target or target-like object exists
  simultaneously with the original or its continuation.

The duplicate may:
- look almost identical to the original,
- be slightly mutated,
- be a similar object rather than an exact copy,
- become the object the robot manipulates afterward.

IMPORTANT:
- A similar object that was already present in SOURCE/BEFORE is NOT new duplication.
- Occlusion is not duplication.
- Appearance change of one single object is not duplication unless there are
  two simultaneously existing physical identities.
- Compare BEFORE versus AFTER explicitly.

Choose exactly ONE:

A = NO DUPLICATION
B = AMBIGUOUS
C = CLEAR DUPLICATION

Answer exactly one letter: A, B, or C.
"""

DIS_PROMPT = """
You are evaluating UNEXPLAINED TARGET OBJECT DISAPPEARANCE in a generated
robot manipulation video.

INPUT ORDER:
1. SOURCE IMAGE: physical reference scene before generation.
2. BEFORE frames: generated video frames immediately before the suspected event.
3. AFTER frames: generated video frames immediately after the suspected event.

Focus ONLY on the task-relevant target object.

A DISAPPEARANCE violation means:

- The target clearly exists in SOURCE/BEFORE.
- The target then becomes absent in AFTER.
- There is no physically plausible explanation for that absence.

Do NOT count as disappearance if the object is:
- temporarily occluded by robot/gripper,
- clearly placed inside a container,
- naturally moved outside the camera view,
- otherwise physically hidden in a plausible way.

The key question is:
"Was the target visibly present BEFORE and then suddenly absent AFTER
without a valid physical cause?"

Choose exactly ONE:

A = NO UNEXPLAINED DISAPPEARANCE
B = AMBIGUOUS
C = CLEAR UNEXPLAINED DISAPPEARANCE

Answer exactly one letter: A, B, or C.
"""


def token_id(processor, letter):
    for x in [letter, " " + letter]:
        ids = processor.tokenizer.encode(
            x, add_special_tokens=False
        )
        if len(ids) == 1:
            return ids[0]
    raise RuntimeError(letter)


def read_frame(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Cannot read frame {idx}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def label_image(img, text):
    img = img.convert("RGB")

    if img.width > 640:
        scale = 640 / img.width
        img = img.resize(
            (640, int(img.height * scale)),
            Image.Resampling.LANCZOS
        )

    canvas = Image.new(
        "RGB",
        (img.width, img.height + 38),
        "black"
    )
    canvas.paste(img, (0, 38))

    d = ImageDraw.Draw(canvas)
    d.text((8, 10), text, fill="white")

    return canvas


print("Loading Qwen...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

IDA = token_id(processor, "A")
IDB = token_id(processor, "B")
IDC = token_id(processor, "C")


def evaluate(source_path, video_path, before, after, prompt):

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        files = []

        # SOURCE
        img = Image.open(source_path).convert("RGB")
        img = label_image(
            img,
            "SOURCE IMAGE - physical reference"
        )
        p = td / "00_source.jpg"
        img.save(p, quality=95)
        files.append(p)

        # BEFORE
        for k, idx in enumerate(before):
            img = read_frame(video_path, idx)
            img = label_image(
                img,
                f"BEFORE - frame {idx}"
            )

            p = td / f"before_{k}_{idx}.jpg"
            img.save(p, quality=95)
            files.append(p)

        # AFTER
        for k, idx in enumerate(after):
            img = read_frame(video_path, idx)
            img = label_image(
                img,
                f"AFTER - frame {idx}"
            )

            p = td / f"after_{k}_{idx}.jpg"
            img.save(p, quality=95)
            files.append(p)

        content = []

        for p in files:
            content.append({
                "type": "image",
                "image": "file://" + str(p.resolve())
            })

        content.append({
            "type": "text",
            "text": prompt
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

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.inference_mode():
            out = model(**inputs)

        logits = out.logits[0, -1].float()

        abc = torch.stack([
            logits[IDA],
            logits[IDB],
            logits[IDC]
        ])

        probs = torch.softmax(abc, dim=0).cpu().numpy()

        pA, pB, pC = map(float, probs)

        score = 100 * (pC + 0.5 * pB)

        pred = ["A", "B", "C"][
            int(probs.argmax())
        ]

        return pred, score, pA, pB, pC


df = pd.read_csv("positives_v2_manifest.csv")
df = df[df["case"].isin(CASES)].copy()

rows = []

for _, r in df.iterrows():

    case = r["case"]
    cfg = CASES[case]

    print("\n" + "="*90)
    print(case, "| GT:", cfg["gt"])
    print("BEFORE:", cfg["before"])
    print("AFTER :", cfg["after"])

    # Run BOTH branches on every case.
    dup = evaluate(
        r["source_path"],
        r["video_path"],
        cfg["before"],
        cfg["after"],
        DUP_PROMPT
    )

    dis = evaluate(
        r["source_path"],
        r["video_path"],
        cfg["before"],
        cfg["after"],
        DIS_PROMPT
    )

    print(
        f"DUP: pred={dup[0]} score={dup[1]:.2f} "
        f"A={dup[2]:.3f} B={dup[3]:.3f} C={dup[4]:.3f}"
    )

    print(
        f"DIS: pred={dis[0]} score={dis[1]:.2f} "
        f"A={dis[2]:.3f} B={dis[3]:.3f} C={dis[4]:.3f}"
    )

    rows.append({
        "case": case,
        "gt": cfg["gt"],

        "before": ",".join(map(str, cfg["before"])),
        "after": ",".join(map(str, cfg["after"])),

        "dup_pred": dup[0],
        "dup_score": dup[1],
        "dup_pA": dup[2],
        "dup_pB": dup[3],
        "dup_pC": dup[4],

        "dis_pred": dis[0],
        "dis_score": dis[1],
        "dis_pA": dis[2],
        "dis_pB": dis[3],
        "dis_pC": dis[4],
    })


out = pd.DataFrame(rows)

out.to_csv(
    "IDENTITY_V3_ORACLE_SUMMARY.csv",
    index=False
)

print("\n" + "="*90)
print("FINAL")
print("="*90)

print(
    out[[
        "case",
        "gt",
        "dup_pred",
        "dup_score",
        "dis_pred",
        "dis_score"
    ]].to_string(index=False)
)

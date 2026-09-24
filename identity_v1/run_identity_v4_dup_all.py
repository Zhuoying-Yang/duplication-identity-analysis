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

WINDOW = 6          # 每次给 Qwen 6 个连续 generated frames
STRIDE = 4          # 每 4 frames 扫一次
RESIZE_W = 640

CASES = None


def choice_token_id(tokenizer, letter):
    for x in [letter, " " + letter]:
        ids = tokenizer.encode(x, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    raise RuntimeError(f"No single token for {letter}")


def load_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def add_label(img, label):
    img = img.copy().convert("RGB")

    if img.width > RESIZE_W:
        ratio = RESIZE_W / img.width
        img = img.resize(
            (RESIZE_W, int(img.height * ratio)),
            Image.Resampling.LANCZOS
        )

    canvas = Image.new(
        "RGB",
        (img.width, img.height + 34),
        "black"
    )
    canvas.paste(img, (0, 34))

    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), label, fill="white")

    return canvas


print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

print("Loading Qwen3-VL...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

tok = processor.tokenizer

ID_A = choice_token_id(tok, "A")
ID_B = choice_token_id(tok, "B")
ID_C = choice_token_id(tok, "C")
ID_D = choice_token_id(tok, "D")

print("A/B/C/D token ids:", ID_A, ID_B, ID_C, ID_D)


def score_window(source_path, video_path, frame_ids, task_prompt):

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        source = Image.open(source_path).convert("RGB")
        source = add_label(
            source,
            "SOURCE IMAGE — physical reference scene"
        )

        source_file = td / "source.jpg"
        source.save(source_file, quality=95)

        cap = cv2.VideoCapture(video_path)

        generated_files = []

        for k, idx in enumerate(frame_ids):
            img = load_frame(cap, idx)
            if img is None:
                continue

            img = add_label(
                img,
                f"GENERATED VIDEO — frame {idx}"
            )

            p = td / f"gen_{k:02d}_{idx:04d}.jpg"
            img.save(p, quality=92)
            generated_files.append(p)

        cap.release()

        if len(generated_files) < 3:
            raise RuntimeError("Too few generated frames")

        prompt = f"""
You are evaluating OBJECT IDENTITY CONSERVATION in a generated robot
manipulation video.

TASK:
{task_prompt}

IMPORTANT INPUT STRUCTURE:

- The FIRST image is the SOURCE IMAGE.
  It is the physical reference scene before video generation.

- All following images are GENERATED VIDEO frames shown in chronological order.

The generated video's first frame is NOT the physical reference.
You MUST compare the generated frames against the SOURCE IMAGE.

Focus only on the TASK-RELEVANT TARGET OBJECT.

We care about two identity failures:

C = DUPLICATION
A new additional physical instance of the target or a very similar/mutated
version appears in the generated frames compared with the source scene.

Examples:
- source has one white cup, generated scene has two physical cups
- source has one target object, generated scene contains original + a similar
  mutated copy

D = DISAPPEARANCE
The target object that should persist from the source scene becomes absent
without a physically plausible explanation.

Do NOT call it disappearance if the target is:
- temporarily occluded by the robot/gripper
- physically placed inside a container
- naturally moved outside the camera view

Choose exactly ONE:

A = CONSISTENT
No clear identity conservation violation in this local temporal window.

B = AMBIGUOUS
Possible identity violation, but visual evidence is insufficient.

C = DUPLICATION
Clear extra target/target-like physical instance relative to the source scene.

D = DISAPPEARANCE
Clear unexplained loss of the target relative to the source scene.

Judge ONLY from the source image and these generated frames.
Do not invent events outside the shown window.

Answer with exactly one letter: A, B, C, or D.
"""

        content = [
            {
                "type": "image",
                "image": "file://" + str(source_file.resolve())
            }
        ]

        for p in generated_files:
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
        )

        inputs = inputs.to(model.device)

        with torch.inference_mode():
            out = model(**inputs)

        logits = out.logits[0, -1].float()

        vals = torch.stack([
            logits[ID_A],
            logits[ID_B],
            logits[ID_C],
            logits[ID_D]
        ])

        probs = torch.softmax(vals, dim=0).cpu().numpy()

        pA, pB, pC, pD = [float(x) for x in probs]

        pred = ["A", "B", "C", "D"][int(np.argmax(probs))]

        # higher = stronger identity violation evidence
        violation_score = 100.0 * (
            pC + pD + 0.5 * pB
        )

        return {
            "pred": pred,
            "pA": pA,
            "pB": pB,
            "pC_dup": pC,
            "pD_disappear": pD,
            "violation_score": violation_score,
        }


df = pd.read_csv("positives_v2_manifest.csv")
if CASES is not None:
    df = df[df["case"].isin(CASES)].copy()

all_rows = []
summary = []

for _, r in df.iterrows():

    case = r["case"]
    video_path = r["video_path"]
    source_path = r["source_path"]
    task_prompt = r["task_prompt"]

    print("\n" + "=" * 100)
    print(case)
    print("=" * 100)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    starts = list(range(0, total, STRIDE))

    case_rows = []

    for j, start in enumerate(starts):

        # local chronological window around this point
        frame_ids = [
            min(total - 1, start + k)
            for k in range(WINDOW)
        ]

        # remove repeated tail indices
        frame_ids = sorted(set(frame_ids))

        try:
            result = score_window(
                source_path,
                video_path,
                frame_ids,
                task_prompt
            )

            row = {
                "case": case,
                "start": start,
                "end": frame_ids[-1],
                "frames": ",".join(map(str, frame_ids)),
                **result
            }

            case_rows.append(row)
            all_rows.append(row)

            print(
                f"[{j+1:02d}/{len(starts):02d}] "
                f"{frame_ids[0]:03d}-{frame_ids[-1]:03d} "
                f"pred={result['pred']} "
                f"V={result['violation_score']:.1f} "
                f"dup={100*result['pC_dup']:.1f} "
                f"dis={100*result['pD_disappear']:.1f}"
            )

        except Exception as e:
            print("ERROR", start, repr(e))

    cdf = pd.DataFrame(case_rows)

    if len(cdf) == 0:
        continue

    # strongest identity violation
    best = cdf.loc[cdf["violation_score"].idxmax()]

    # strongest subtype-specific evidence
    best_dup = cdf.loc[cdf["pC_dup"].idxmax()]
    best_dis = cdf.loc[cdf["pD_disappear"].idxmax()]

    summary.append({
        "case": case,

        "best_violation_score":
            float(best["violation_score"]),
        "best_violation_pred":
            best["pred"],
        "best_violation_frames":
            best["frames"],

        "max_dup_score":
            100 * float(best_dup["pC_dup"]),
        "max_dup_frames":
            best_dup["frames"],

        "max_disappear_score":
            100 * float(best_dis["pD_disappear"]),
        "max_disappear_frames":
            best_dis["frames"],
    })

    pd.DataFrame(all_rows).to_csv(
        "IDENTITY_V4_DUP_ALL_WINDOWS.csv",
        index=False
    )

    pd.DataFrame(summary).to_csv(
        "IDENTITY_V4_DUP_SUMMARY.csv",
        index=False
    )


print("\n" + "=" * 100)
print("FINAL SUMMARY")
print("=" * 100)

s = pd.DataFrame(summary)
print(s.to_string(index=False))

s.to_csv(
    "IDENTITY_V4_DUP_SUMMARY.csv",
    index=False
)

pd.DataFrame(all_rows).to_csv(
    "IDENTITY_V4_DUP_ALL_WINDOWS.csv",
    index=False
)

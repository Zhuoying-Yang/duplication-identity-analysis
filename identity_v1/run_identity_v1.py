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

N_FRAMES = 36
FRAMES_PER_GRID = 6


PROMPT = """
You are evaluating object identity consistency in a generated
robot manipulation video.

Focus on the SINGLE target object that the robot is intended to manipulate.

A physically consistent target should preserve one continuous physical
identity throughout the video.

A CLEAR IDENTITY VIOLATION occurs if:

1. DUPLICATION:
   One target object spontaneously becomes two simultaneously existing
   physical instances.

   The new instance may look exactly the same as the original, or may
   look like a similar / mutated object.

2. DISAPPEARANCE:
   The target object unexpectedly disappears without a physically
   plausible explanation.

Important:

- Multiple similar objects already present from the beginning are NOT duplication.
- Duplication requires a NEW additional object instance to appear.
- Temporary occlusion by the robot or gripper is NOT disappearance.
- Putting the object inside a drawer/container is NOT disappearance.
- Naturally leaving the camera view is NOT disappearance.
- The duplicate may look slightly different from the original.
- Judge the temporal sequence across all images.

Choose exactly one:

A = CONSISTENT
    The target remains one physically coherent object.

B = AMBIGUOUS
    There may be an identity violation, but the evidence is unclear.

C = CLEAR IDENTITY VIOLATION
    Clear spontaneous 1->2 duplication or unexplained 1->0 disappearance.

Answer with exactly one letter: A, B, or C.
"""


def read_uniform_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Cannot read video: {video_path}")

    inds = np.linspace(
        0,
        total - 1,
        min(N_FRAMES, total)
    ).round().astype(int)

    frames = []

    for idx in inds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()

        if not ok:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append((int(idx), Image.fromarray(frame)))

    cap.release()

    if len(frames) < 4:
        raise RuntimeError(f"Only {len(frames)} readable frames")

    return frames


def make_grid(group, save_path):
    cols = 3
    rows = 2

    W = 320
    H = 200

    canvas = Image.new(
        "RGB",
        (cols * W, rows * H),
        "black"
    )

    draw = ImageDraw.Draw(canvas)

    for k, (frame_idx, img) in enumerate(group):
        r = k // cols
        c = k % cols

        img = img.copy()
        img.thumbnail((W, H))

        x = c * W + (W - img.width) // 2
        y = r * H + (H - img.height) // 2

        canvas.paste(img, (x, y))

        draw.rectangle(
            [
                c * W,
                r * H,
                c * W + 110,
                r * H + 24
            ],
            fill="black"
        )

        draw.text(
            (c * W + 5, r * H + 4),
            f"frame {frame_idx}",
            fill="white"
        )

    canvas.save(save_path)


def make_grids(video_path, tmpdir):
    frames = read_uniform_frames(video_path)

    paths = []

    for i in range(0, len(frames), FRAMES_PER_GRID):
        group = frames[i:i + FRAMES_PER_GRID]

        p = Path(tmpdir) / f"grid_{i // FRAMES_PER_GRID:02d}.jpg"

        make_grid(group, p)
        paths.append(str(p))

    return paths


print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

print("Loading Qwen3-VL...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model.eval()


def token_id(letter):
    for x in [letter, " " + letter]:
        ids = processor.tokenizer.encode(
            x,
            add_special_tokens=False
        )

        if len(ids) == 1:
            return ids[0]

    raise RuntimeError(f"No single token for {letter}")


ID_A = token_id("A")
ID_B = token_id("B")
ID_C = token_id("C")

print("A/B/C token ids:", ID_A, ID_B, ID_C)


def score_video(video_path):

    with tempfile.TemporaryDirectory() as td:

        grid_paths = make_grids(
            video_path,
            td
        )

        content = []

        for p in grid_paths:
            content.append({
                "type": "image",
                "image": "file://" + str(Path(p).resolve())
            })

        content.append({
            "type": "text",
            "text": PROMPT
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

        abc_logits = torch.stack([
            logits[ID_A],
            logits[ID_B],
            logits[ID_C]
        ])

        probs = torch.softmax(
            abc_logits,
            dim=0
        ).cpu().numpy()

        pA, pB, pC = [float(x) for x in probs]

        score = 100 * (
            pC + 0.5 * pB
        )

        pred = ["A", "B", "C"][
            int(np.argmax(probs))
        ]

        return {
            "pred": pred,
            "pA": pA,
            "pB": pB,
            "pC": pC,
            "identity_violation_score": score,
            "n_grids": len(grid_paths)
        }


df = pd.read_csv("manifest.csv")

rows = []

for i, r in df.iterrows():

    case = r["case"]
    video_path = r["video_path"]

    print()
    print("=" * 80)
    print(f"[{i+1}/{len(df)}] {case}")
    print(video_path)

    try:
        result = score_video(video_path)

        print(
            f"pred={result['pred']} "
            f"score={result['identity_violation_score']:.3f} "
            f"A={result['pA']:.3f} "
            f"B={result['pB']:.3f} "
            f"C={result['pC']:.3f}"
        )

        row = {
            **r.to_dict(),
            **result
        }

    except Exception as e:

        print("ERROR:", repr(e))

        row = {
            **r.to_dict(),
            "pred": "ERROR",
            "pA": np.nan,
            "pB": np.nan,
            "pC": np.nan,
            "identity_violation_score": np.nan,
            "n_grids": np.nan
        }

    rows.append(row)

    pd.DataFrame(rows).to_csv(
        "IDENTITY_V1_RESULTS.csv",
        index=False
    )


print()
print("=" * 80)
print("DONE")
print("=" * 80)

out = pd.DataFrame(rows)

print(
    out[
        [
            "case",
            "pred",
            "identity_violation_score",
            "pA",
            "pB",
            "pC"
        ]
    ].to_string(index=False)
)

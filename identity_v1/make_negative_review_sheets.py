import cv2
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

CSV = "NEGATIVE_CANDIDATE_POOL_FROZEN.csv"

SOURCE_DIR = Path(
    "/shared/ssd_30T/zhuoyingyang/physact/"
    "evaluation_results/robowm_lvp_68_numbered/images"
)

OUT = Path("negative_review")
OUT.mkdir(exist_ok=True)

STEP = 3
COLS = 4
ROWS = 4
GEN_PER_SHEET = COLS * ROWS - 1   # 15 generated + 1 source

TILE_W = 320
TILE_H = 215
LABEL_H = 28


def case_index(case):
    return int(case.split("_")[-1])


def fit_image(img):
    img = img.convert("RGB")
    img.thumbnail((TILE_W, TILE_H))

    canvas = Image.new(
        "RGB",
        (TILE_W, TILE_H),
        "black"
    )

    canvas.paste(
        img,
        (
            (TILE_W - img.width)//2,
            (TILE_H - img.height)//2
        )
    )

    return canvas


def tile(img, label):
    img = fit_image(img)

    canvas = Image.new(
        "RGB",
        (TILE_W, TILE_H + LABEL_H),
        "black"
    )

    canvas.paste(img, (0, LABEL_H))

    d = ImageDraw.Draw(canvas)
    d.text((7, 7), label, fill="white")

    return canvas


df = pd.read_csv(CSV)

for n, r in df.iterrows():

    case = r["case"]
    video = Path(r["video_path"])
    idx = case_index(case)

    case_dir = OUT / case
    case_dir.mkdir(parents=True, exist_ok=True)

    source_path = SOURCE_DIR / f"{idx:04d}.png"

    if not source_path.exists():
        print("SOURCE MISSING:", case, source_path)
        continue

    source_img = Image.open(source_path).convert("RGB")

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_ids = list(range(0, total, STEP))

    # always include last frame
    if total > 0 and (total - 1) not in frame_ids:
        frame_ids.append(total - 1)

    frames = {}

    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)

        ok, frame = cap.read()

        if not ok:
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frames[fid] = Image.fromarray(frame)

    cap.release()

    valid_ids = list(frames.keys())

    chunks = [
        valid_ids[i:i+GEN_PER_SHEET]
        for i in range(0, len(valid_ids), GEN_PER_SHEET)
    ]

    for si, ids in enumerate(chunks):

        tiles = [
            tile(
                source_img,
                f"SOURCE #{idx:04d}"
            )
        ]

        for fid in ids:
            tiles.append(
                tile(
                    frames[fid],
                    f"FRAME {fid}"
                )
            )

        while len(tiles) < COLS * ROWS:
            tiles.append(
                Image.new(
                    "RGB",
                    (TILE_W, TILE_H + LABEL_H),
                    "black"
                )
            )

        sheet = Image.new(
            "RGB",
            (
                COLS * TILE_W,
                ROWS * (TILE_H + LABEL_H)
            ),
            "black"
        )

        for i, t in enumerate(tiles):
            x = (i % COLS) * TILE_W
            y = (i // COLS) * (TILE_H + LABEL_H)
            sheet.paste(t, (x, y))

        out = case_dir / f"sheet_{si+1:02d}.jpg"
        sheet.save(out, quality=94)

    print(
        f"[{n+1:02d}/{len(df):02d}]",
        case,
        "frames=", total,
        "sheets=", len(chunks)
    )


# annotation template
ann = df.copy()

ann["human_identity_gt"] = ""
ann["human_note"] = ""

ann.to_csv(
    "NEGATIVE40_HUMAN_ANNOTATION.csv",
    index=False
)

print()
print("DONE")
print("Review folder:", OUT)
print("Annotation CSV: NEGATIVE40_HUMAN_ANNOTATION.csv")

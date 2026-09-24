from pathlib import Path
import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

BASE = Path("/shared/ssd_30T/zhuoyingyang/physact/sam3_robowm/identity_v1")

manifest = pd.read_csv(BASE/"EXPANSION6_FROZEN.csv")
cand = pd.read_csv(BASE/"sam_masked_expansion6/SAM_EXPANSION6_CANDIDATES.csv")
frames = pd.read_csv(BASE/"dinov2_expansion6_frozen/DINOV2_EXPANSION6_FRAMES.csv")

OUT = BASE/"expansion6_visual_audit"
OUT.mkdir(exist_ok=True)

AUDIT = {
    "Cosmos2.5_0058":[0,12,15,24,27,84,87,92],
    "Cosmos2.5_0060":[0,3,6,12,18,27,45,92],
    "Cosmos3_seed101_0010":[0,6,33,60,72,90,93,132,156,188],
    "Cosmos3_seed101_0020":[0,117,120,126,129,141,174,186,188],
    "Cosmos3_seed101_0036":[0,30,93,123,132,135,171,183,188],
    "LVP_0030":[0,15,18,24,39,42,45,48],
}

for _, cfg in manifest.iterrows():
    case = cfg["case"]

    if case not in AUDIT:
        continue

    case_out = OUT/case
    case_out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(cfg["video_path"]))

    for frame in AUDIT[case]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok,bgr = cap.read()
        if not ok:
            continue

        rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        full = Image.fromarray(rgb).convert("RGB")
        draw = ImageDraw.Draw(full)

        g = cand[
            (cand["case"]==case) &
            (cand["frame"]==frame)
        ].sort_values("rank")

        for _,r in g.iterrows():
            rank=int(r["rank"])
            box=(r["x1"],r["y1"],r["x2"],r["y2"])
            draw.rectangle(box,width=4)
            draw.text(
                (r["x1"]+4,r["y1"]+4),
                f"R{rank}",
                fill="white",
                stroke_width=2,
                stroke_fill="black"
            )

        ff = frames[
            (frames["case"]==case) &
            (frames["frame"]==frame)
        ]

        if len(ff):
            score=float(ff.iloc[0]["frame_score"])
            pair=str(ff.iloc[0]["best_pair"])
        else:
            score=0.0
            pair=""

        thumb_size=220
        canvas=Image.new(
            "RGB",
            (
                full.width + thumb_size,
                max(full.height, thumb_size*3+50)
            ),
            "white"
        )

        canvas.paste(full,(0,50))

        cd=ImageDraw.Draw(canvas)
        cd.text(
            (10,10),
            f"{case}  frame={frame}  DINO={score:.3f}  best={pair}",
            fill="black"
        )

        for k,(_,r) in enumerate(g.head(3).iterrows()):
            crop=Image.open(r["crop_path"]).convert("RGB")
            crop.thumbnail((thumb_size-20,thumb_size-35))

            x=full.width+10
            y=50+k*thumb_size

            canvas.paste(crop,(x,y+20))
            cd.text(
                (x,y),
                f"R{int(r['rank'])}",
                fill="black"
            )

        canvas.save(
            case_out/f"frame_{frame:04d}.jpg",
            quality=95
        )

    cap.release()

print("DONE:",OUT)

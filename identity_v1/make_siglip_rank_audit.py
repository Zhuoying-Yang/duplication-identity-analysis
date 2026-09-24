from pathlib import Path
import math

import cv2
import pandas as pd
from PIL import Image, ImageDraw


OUT = Path("siglip_rank_visual_audit")
OUT.mkdir(exist_ok=True)

manifest = pd.read_csv("IDENTITY_EVAL_V1.csv")

sig = pd.read_csv(
    "siglip_identity_eval_v1/"
    "SIGLIP_IDENTITY_EVAL_V1_CANDIDATES.csv"
)

pre = pd.read_csv(
    "IDENTITY_PRETRACK_DIAGNOSTIC.csv"
)


CASES = [
    "Cosmos3_seed102_0010",
    "Cosmos3_seed102_0006",
    "Cosmos3_seed103_0012",
    "Cosmos3_seed102_0008",
]


def get_frame(video, idx):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def make_sheet(imgs, titles, out):
    if not imgs:
        return

    W,H = 700,440
    cells=[]

    for img,title in zip(imgs,titles):

        rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        im=Image.fromarray(rgb)
        im.thumbnail((W,H-45))

        cell=Image.new("RGB",(W,H),"white")
        cell.paste(im,((W-im.width)//2,40))

        d=ImageDraw.Draw(cell)
        d.text((8,8),title,fill="black")

        cells.append(cell)

    cols=2
    rows=math.ceil(len(cells)/cols)

    sheet=Image.new(
        "RGB",
        (cols*W,rows*H),
        "white"
    )

    for i,c in enumerate(cells):
        sheet.paste(
            c,
            ((i%cols)*W,(i//cols)*H)
        )

    sheet.save(out,quality=95)


for case in CASES:

    print("\n",case)

    mr=manifest[manifest.case==case].iloc[0]
    video=mr.video_path

    q=sig[sig.case==case].copy()

    frames=sorted(q.frame.unique())

    # --------------------------------------------------
    # Show 16 frames distributed through the video.
    # This avoids selecting frames using evaluator score.
    # --------------------------------------------------

    if len(frames)<=16:
        selected=frames
    else:
        inds=[
            round(i*(len(frames)-1)/15)
            for i in range(16)
        ]
        selected=[frames[i] for i in inds]


    imgs=[]
    titles=[]

    for fr in selected:

        img=get_frame(video,fr)

        if img is None:
            continue

        fq=(
            q[q.frame==fr]
            .sort_values("rank")
            .head(12)
        )

        for _,r in fq.iterrows():

            rank=int(r["rank"])

            x1,y1,x2,y2=map(
                int,
                [r.x1,r.y1,r.x2,r.y2]
            )

            # Top3 = thicker box.
            thickness=4 if rank<=3 else 2

            # No special colors needed:
            # label itself tells us rank.
            cv2.rectangle(
                img,
                (x1,y1),
                (x2,y2),
                (255,255,255),
                thickness
            )

            text=(
                f"R{rank} "
                f"S={r.source_similarity:.2f} "
                f"G={r.gdino_score:.2f}"
            )

            cv2.putText(
                img,
                text,
                (x1,max(18,y1-5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255,255,255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                img,
                text,
                (x1,max(18,y1-5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0,0,0),
                1,
                cv2.LINE_AA
            )

        imgs.append(img)

        titles.append(
            f"{case}  frame={int(fr)}  "
            f"proposals={len(fq)}"
        )


    save=OUT/f"{case}_SIGLIP_RANKS.jpg"

    make_sheet(
        imgs,
        titles,
        save
    )

    print("saved:",save)


print("\nDONE")

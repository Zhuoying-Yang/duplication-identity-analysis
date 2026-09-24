import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


OUT = Path("identity_falsepair_visual_audit")
OUT.mkdir(exist_ok=True)

manifest = pd.read_csv("IDENTITY_EVAL_V1.csv")

cand = pd.read_csv(
    "sam_identity_eval_v1/"
    "SAM_IDENTITY_EVAL_V1_CANDIDATES.csv"
)

pairs = pd.read_csv(
    "dinov2_identity_eval_v1/"
    "DINOV2_IDENTITY_EVAL_V1_PAIRS.csv"
)


CASES = [
    # strong true positives
    "Cosmos3_seed102_0010",
    "Cosmos3_seed102_0006",
    "Cosmos3_seed103_0012",
    "Cosmos3_seed102_0008",

    # strongest clean false positives
    "Cosmos3_seed102_0009",
    "Cosmos3_seed103_0008",
    "Cosmos3_seed102_0011",
    "Cosmos3_seed103_0009",
    "Cosmos3_seed103_0011",

    # zero-pair positive cases
    "Cosmos3_seed102_0062",
    "Cosmos3_seed103_0058",
    "Cosmos3_seed103_0062",
]


# ============================================================
# Evidence
# ============================================================

p = pairs.copy()

if "nested" in p.columns:
    if p["nested"].dtype == bool:
        nested = p["nested"]
    else:
        nested = (
            p["nested"]
            .astype(str)
            .str.lower()
            .isin(["true","1","yes"])
        )

    p = p[~nested].copy()


p["min_gdino"] = p[
    ["gdino_a","gdino_b"]
].min(axis=1)

p["min_siglip"] = p[
    ["siglip_source_a","siglip_source_b"]
].min(axis=1)

p["semantic_support"] = np.sqrt(
    p["min_gdino"].clip(lower=0)
    *
    p["min_siglip"].clip(lower=0)
)

p["evidence"] = (
    p["pair_similarity"]
    *
    p["semantic_support"]
)


# ============================================================
# Helpers
# ============================================================

def get_frame(video_path, frame_idx):

    cap = cv2.VideoCapture(video_path)

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(frame_idx)
    )

    ok, frame = cap.read()

    cap.release()

    if not ok:
        return None

    return frame


def draw_candidate(
    img,
    r,
    selected=False
):

    x1,y1,x2,y2 = map(
        int,
        [r.x1,r.y1,r.x2,r.y2]
    )

    color = (
        (0,0,255)
        if selected
        else (255,0,0)
    )

    thickness = 4 if selected else 2

    cv2.rectangle(
        img,
        (x1,y1),
        (x2,y2),
        color,
        thickness
    )

    text = (
        f"R{int(r['rank'])} "
        f"G={float(r.gdino_score):.2f} "
        f"S={float(r.source_similarity):.2f}"
    )

    cv2.putText(
        img,
        text,
        (x1,max(20,y1-7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA
    )


def make_sheet(images, titles, save_path):

    if not images:
        return

    target_w = 520
    target_h = 340

    cells = []

    for img,title in zip(images,titles):

        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        pil = Image.fromarray(rgb)

        pil.thumbnail(
            (target_w,target_h-40)
        )

        cell = Image.new(
            "RGB",
            (target_w,target_h),
            "white"
        )

        x = (
            target_w-pil.width
        )//2

        cell.paste(
            pil,
            (x,35)
        )

        d = ImageDraw.Draw(cell)

        d.text(
            (8,8),
            title,
            fill="black"
        )

        cells.append(cell)


    cols = 2
    rows = math.ceil(len(cells)/cols)

    sheet = Image.new(
        "RGB",
        (
            cols*target_w,
            rows*target_h
        ),
        "white"
    )

    for i,c in enumerate(cells):

        sheet.paste(
            c,
            (
                (i%cols)*target_w,
                (i//cols)*target_h
            )
        )

    sheet.save(
        save_path,
        quality=95
    )


# ============================================================
# Cases
# ============================================================

for case in CASES:

    print("\n",case)

    mr = manifest[
        manifest.case == case
    ]

    if mr.empty:
        print("  missing manifest")
        continue

    video_path = mr.iloc[0]["video_path"]

    pc = p[
        p.case == case
    ].copy()

    cc = cand[
        cand.case == case
    ].copy()


    # --------------------------------------------------------
    # Normal case: top evidence frames
    # --------------------------------------------------------

    if len(pc):

        best = (
            pc.sort_values(
                "evidence",
                ascending=False
            )
            .groupby(
                "frame",
                as_index=False
            )
            .first()
            .sort_values(
                "evidence",
                ascending=False
            )
            .head(6)
        )

        images = []
        titles = []

        for _,r in best.iterrows():

            fr = int(r.frame)

            img = get_frame(
                video_path,
                fr
            )

            if img is None:
                continue

            fc = cc[
                cc.frame == fr
            ]

            for _,cr in fc.iterrows():

                selected = (
                    int(cr["rank"])
                    in [
                        int(r.rank_a),
                        int(r.rank_b)
                    ]
                )

                draw_candidate(
                    img,
                    cr,
                    selected
                )

            title = (
                f"f{fr} "
                f"E={r.evidence:.3f} "
                f"DINO={r.pair_similarity:.3f} "
                f"Gmin={r.min_gdino:.3f}"
            )

            images.append(img)
            titles.append(title)


    # --------------------------------------------------------
    # No valid DINO pair:
    # show frames with most SAM candidates
    # --------------------------------------------------------

    else:

        counts = (
            cc.groupby("frame")
            .size()
            .reset_index(name="n")
            .sort_values(
                ["n","frame"],
                ascending=[False,True]
            )
            .head(6)
        )

        images = []
        titles = []

        for _,r in counts.iterrows():

            fr = int(r.frame)

            img = get_frame(
                video_path,
                fr
            )

            if img is None:
                continue

            fc = cc[
                cc.frame == fr
            ]

            for _,cr in fc.iterrows():
                draw_candidate(
                    img,
                    cr,
                    False
                )

            images.append(img)

            titles.append(
                f"f{fr} "
                f"SAM candidates={len(fc)} "
                f"NO VALID NONNESTED PAIR"
            )


    save = (
        OUT /
        f"{case}_PAIR_AUDIT.jpg"
    )

    make_sheet(
        images,
        titles,
        save
    )

    print(" saved:",save)


print("\nDONE:",OUT)

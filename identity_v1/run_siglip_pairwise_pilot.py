from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipModel

ROOT = Path("/shared/ssd_30T/zhuoyingyang/physact")

IN = Path(
    "siglip_source_match_pilot/SIGLIP_CANDIDATES.csv"
)

OUT = Path(
    "siglip_source_match_pilot/"
    "SIGLIP_PAIRWISE_FRAME_SUMMARY.csv"
)

SIGLIP_CACHE = (
    ROOT
    / "cosmos_predict25/hf_cache/hub/"
    "models--google--siglip-so400m-patch14-384"
)

snapshots = sorted(
    (SIGLIP_CACHE / "snapshots").glob("*")
)

SIGLIP_PATH = snapshots[-1]

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

CROP_EXPAND = 0.08

device = "cuda" if torch.cuda.is_available() else "cpu"

print("device:", device)
print("SigLIP:", SIGLIP_PATH)

processor = AutoImageProcessor.from_pretrained(
    str(SIGLIP_PATH),
    local_files_only=True,
)

model = SiglipModel.from_pretrained(
    str(SIGLIP_PATH),
    local_files_only=True,
).to(device).eval()

print("SigLIP ready.")


@torch.no_grad()
def embed(images):
    inp = processor(
        images=images,
        return_tensors="pt"
    )

    feats = model.get_image_features(
        pixel_values=inp["pixel_values"].to(device)
    )

    feats = feats / (
        feats.norm(dim=-1, keepdim=True) + 1e-8
    )

    return feats


def crop_box(rgb, r):
    H, W = rgb.shape[:2]

    x1 = float(r.x1)
    y1 = float(r.y1)
    x2 = float(r.x2)
    y2 = float(r.y2)

    bw = x2-x1
    bh = y2-y1

    x1 -= bw*CROP_EXPAND
    x2 += bw*CROP_EXPAND
    y1 -= bh*CROP_EXPAND
    y2 += bh*CROP_EXPAND

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(W, int(x2))
    y2 = min(H, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return Image.fromarray(
        rgb[y1:y2, x1:x2]
    )


d = pd.read_csv(IN)

rows = []

for case in d.case.unique():

    print("\n"+"="*80)
    print(case)
    print("="*80)

    cap = cv2.VideoCapture(
        str(VIDEOS[case])
    )

    z = d[d.case == case]

    for frame, g in z.groupby("frame"):

        frame = int(frame)

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame
        )

        ok, bgr = cap.read()

        if not ok:
            continue

        rgb = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB
        )

        valid_rows = []
        crops = []

        for r in g.itertuples():

            c = crop_box(rgb, r)

            if c is None:
                continue

            valid_rows.append(r)
            crops.append(c)

        if len(crops) < 2:

            rows.append({
                "case": case,
                "frame": frame,
                "best_pair_score": 0.0,
                "pair_similarity": np.nan,
                "min_source_similarity": np.nan,
                "source_sim_a": np.nan,
                "source_sim_b": np.nan,
            })

            continue

        feats = embed(crops)

        pair_matrix = (
            feats @ feats.T
        ).detach().cpu().numpy()

        source_sims = np.array([
            float(r.source_similarity)
            for r in valid_rows
        ])

        best = None

        for i in range(len(crops)):
            for j in range(i+1, len(crops)):

                pair_sim = float(
                    pair_matrix[i,j]
                )

                min_source = min(
                    source_sims[i],
                    source_sims[j]
                )

                # Both must resemble source,
                # AND resemble each other.
                score = (
                    min_source
                    * max(0.0, pair_sim)
                )

                q = {
                    "score": score,
                    "pair_sim": pair_sim,
                    "min_source": min_source,
                    "a": source_sims[i],
                    "b": source_sims[j],
                }

                if (
                    best is None
                    or q["score"] > best["score"]
                ):
                    best = q

        rows.append({
            "case": case,
            "frame": frame,
            "best_pair_score":
                best["score"],
            "pair_similarity":
                best["pair_sim"],
            "min_source_similarity":
                best["min_source"],
            "source_sim_a":
                best["a"],
            "source_sim_b":
                best["b"],
        })

        print(
            f"frame={frame:03d} "
            f"PAIR={best['score']:.3f} "
            f"pairSim={best['pair_sim']:.3f} "
            f"minSrc={best['min_source']:.3f}"
        )

    cap.release()


out = pd.DataFrame(rows)

out.to_csv(
    OUT,
    index=False
)

print("\nSAVED:", OUT)


# ------------------------------------------------------------
# Change-point diagnostic
# ------------------------------------------------------------

W = 5

for case in out.case.unique():

    x = (
        out[out.case == case]
        .sort_values("frame")
        .reset_index(drop=True)
    )

    s = x.best_pair_score.fillna(0).values
    f = x.frame.values

    rr = []

    for i in range(W, len(x)-W):

        pre = np.median(
            s[i-W:i]
        )

        post = np.median(
            s[i:i+W]
        )

        jump = max(
            0.0,
            post-pre
        )

        score = jump*post

        rr.append({
            "frame": int(f[i]),
            "pre": pre,
            "post": post,
            "jump": jump,
            "dup_score": score,
        })

    r = pd.DataFrame(rr)

    best = r.loc[
        r.dup_score.idxmax()
    ]

    print("\n"+"="*80)
    print(case)
    print("="*80)

    print(
        f"BEST FRAME = {int(best.frame)}\n"
        f"pre        = {best.pre:.3f}\n"
        f"post       = {best.post:.3f}\n"
        f"jump       = {best.jump:.3f}\n"
        f"DUP SCORE  = {best.dup_score:.3f}"
    )

    print(
        r.nlargest(
            5,
            "dup_score"
        ).to_string(index=False)
    )

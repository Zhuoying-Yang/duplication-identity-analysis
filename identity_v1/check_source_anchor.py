from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model

BASE = Path(
    "/shared/ssd_30T/zhuoyingyang/physact/"
    "sam3_robowm/identity_v1"
)

CAND = (
    BASE
    / "sam_masked_expansion6"
    / "SAM_EXPANSION6_CANDIDATES.csv"
)

SOURCE = (
    BASE
    / "sam_masked_expansion6"
    / "source"
)

CACHE = (
    "/shared/ssd_30T/zhuoyingyang/"
    "models/hf_cache"
)

MODEL = "facebook/dinov2-base"

d = pd.read_csv(CAND)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

processor = AutoImageProcessor.from_pretrained(
    MODEL,
    cache_dir=CACHE,
    local_files_only=True,
)

model = Dinov2Model.from_pretrained(
    MODEL,
    cache_dir=CACHE,
    local_files_only=True,
).to(device).eval()


@torch.no_grad()
def embed(images):
    x = processor(
        images=images,
        return_tensors="pt"
    )

    y = model(
        pixel_values=x["pixel_values"].to(device)
    )

    f = y.last_hidden_state[:,0]

    f = f / (
        f.norm(
            dim=-1,
            keepdim=True
        ) + 1e-8
    )

    return f


print("="*100)
print("SOURCE → INITIAL FRAME DINO ANCHOR")
print("="*100)

for case in d["case"].unique():

    z = d[
        d["case"] == case
    ].copy()

    first_frame = int(
        z["frame"].min()
    )

    g = (
        z[z["frame"] == first_frame]
        .sort_values("rank")
    )

    source_path = (
        SOURCE / f"{case}.png"
    )

    source_img = Image.open(
        source_path
    ).convert("RGB")

    candidate_imgs = [
        Image.open(p).convert("RGB")
        for p in g["crop_path"]
    ]

    feats = embed(
        [source_img] + candidate_imgs
    )

    src = feats[0]

    sims = (
        feats[1:] @ src
    ).detach().cpu().numpy()

    print()
    print(
        case,
        " first_frame=",
        first_frame
    )

    result = []

    for (_,r),sim in zip(
        g.iterrows(),
        sims
    ):
        result.append({
            "rank":
                int(r["rank"]),

            "source_DINO":
                float(sim),

            "SigLIP":
                float(
                    r["source_similarity"]
                ),

            "GDINO":
                float(
                    r["gdino_score"]
                ),
        })

    q = pd.DataFrame(result)

    print(
        q.to_string(
            index=False
        )
    )

    best = q.loc[
        q["source_DINO"].idxmax()
    ]

    print(
        "ANCHOR => R%d  DINO=%.3f"
        % (
            int(best["rank"]),
            best["source_DINO"]
        )
    )

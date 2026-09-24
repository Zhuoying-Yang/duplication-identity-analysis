import pandas as pd
from pathlib import Path

df = pd.read_csv("positives.csv")

ROOT = Path("/shared/ssd_30T/zhuoyingyang/physact")

BASE = {
    "Cosmos2.5":
        ROOT / "cosmos_predict25/native_outputs/robowm_68_full",

    "Cosmos3_seed101":
        ROOT / "cosmos3/export_robowm68_3seeds/seed101",

    "LVP":
        ROOT / "evaluation_results/robowm_lvp_68_numbered/videos",
}

def resolve(case, dataset):
    idx = case.split("_")[-1]

    if dataset not in BASE:
        print("UNKNOWN DATASET:", dataset)
        return ""

    p = BASE[dataset] / f"{idx}.mp4"

    if not p.exists():
        print("MISSING:", case, "->", p)
        return ""

    return str(p)

df["video_path"] = [
    resolve(r.case, r.dataset)
    for r in df.itertuples()
]

df.to_csv("positives_resolved.csv", index=False)

print(
    df[["case", "dataset", "video_path"]]
    .to_string(index=False)
)

print()
print(
    "Resolved:",
    (df.video_path != "").sum(),
    "/",
    len(df)
)

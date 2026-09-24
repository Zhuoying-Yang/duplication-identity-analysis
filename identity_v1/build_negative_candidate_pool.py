import random
from pathlib import Path
import pandas as pd

SEED = 20260827
rng = random.Random(SEED)

ROOTS = {
    "Cosmos2.5":
        Path("/shared/ssd_30T/zhuoyingyang/physact/cosmos_predict25/native_outputs/robowm_68_full"),

    "Cosmos3_seed101":
        Path("/shared/ssd_30T/zhuoyingyang/physact/cosmos3/export_robowm68_3seeds/seed101"),

    "LVP":
        Path("/shared/ssd_30T/zhuoyingyang/physact/evaluation_results/robowm_lvp_68_numbered/videos"),
}

N_SAMPLE = {
    "Cosmos2.5": 8,
    "Cosmos3_seed101": 20,
    "LVP": 12,
}

pos = pd.read_csv("positives_v2_manifest.csv")

positive_cases = set(pos["case"].astype(str))

rows = []

for dataset, root in ROOTS.items():

    candidates = []

    for p in sorted(root.glob("*.mp4")):

        idx = int(p.stem)

        if dataset == "Cosmos2.5":
            case = f"Cosmos2.5_{idx:04d}"

        elif dataset == "Cosmos3_seed101":
            case = f"Cosmos3_seed101_{idx:04d}"

        else:
            case = f"LVP_{idx:04d}"

        if case in positive_cases:
            continue

        candidates.append((case, str(p)))

    chosen = rng.sample(
        candidates,
        min(N_SAMPLE[dataset], len(candidates))
    )

    for case, path in chosen:
        rows.append({
            "case": case,
            "dataset": dataset,
            "video_path": path,
            "human_identity_gt": "",
            "human_note": "",
        })

df = pd.DataFrame(rows)

df.to_csv(
    "NEGATIVE_CANDIDATE_POOL_FROZEN.csv",
    index=False
)

print("SEED =", SEED)
print()
print(df[["case", "dataset"]].to_string(index=False))

print()
print(df.groupby("dataset").size())
print()
print("TOTAL =", len(df))

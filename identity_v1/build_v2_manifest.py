import json
import pandas as pd
from pathlib import Path

ROOT = Path("/shared/ssd_30T/zhuoyingyang/physact")

SOURCE_DIR = (
    ROOT /
    "evaluation_results/robowm_lvp_68_numbered/images"
)

PROMPT_FILE = (
    ROOT /
    "ReVidgen/data/lvp_robowm68_common/prompts.json"
)

df = pd.read_csv("positives_resolved.csv")
prompts = json.load(open(PROMPT_FILE))

source_paths = []
task_prompts = []
source_ok = []

for _, r in df.iterrows():

    # e.g. Cosmos2.5_0016 -> 16
    idx = int(str(r["case"]).split("_")[-1])

    source = SOURCE_DIR / f"{idx:04d}.png"

    if idx >= len(prompts):
        prompt = ""
        print("PROMPT MISSING:", r["case"])
    else:
        prompt = prompts[idx]["prompt"]

    source_paths.append(str(source))
    task_prompts.append(prompt)
    source_ok.append(source.exists())

df["source_path"] = source_paths
df["task_prompt"] = task_prompts
df["source_exists"] = source_ok

df.to_csv(
    "positives_v2_manifest.csv",
    index=False
)

print(
    df[
        [
            "case",
            "source_exists",
            "source_path",
            "task_prompt"
        ]
    ].to_string(index=False)
)

print()
print(
    "Source images:",
    sum(source_ok),
    "/",
    len(df)
)

print(
    "Prompts:",
    sum(bool(x) for x in task_prompts),
    "/",
    len(df)
)

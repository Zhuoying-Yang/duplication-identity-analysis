import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path("/shared/ssd_30T/zhuoyingyang/physact")

PROMPTS = ROOT / "ReVidgen/data/lvp_robowm68_common/prompts.json"
SOURCE_DIR = ROOT / "evaluation_results/robowm_lvp_68_numbered/images"

COS25 = ROOT / "cosmos_predict25/native_outputs/robowm_68_full"
COS3  = ROOT / "cosmos3/export_robowm68_3seeds"
LVP   = ROOT / "evaluation_results/robowm_lvp_68_numbered/videos"

prompts = json.load(open(PROMPTS))


# ============================================================
# ALL KNOWN DUPLICATION POSITIVES
# ============================================================

positives = {

    "Cosmos2.5": [
        16, 58, 60, 64
    ],

    "Cosmos3_seed101": [
        3, 10, 12, 17, 20,
        27, 36, 54, 55, 65
    ],

    "LVP": [
        4, 30, 53, 54, 55, 56
    ],

    "Cosmos3_seed102": [
        6, 8, 10, 62, 64
    ],

    "Cosmos3_seed103": [
        6, 10, 12, 13, 17,
        56, 58, 60, 62, 64
    ],
}


# ============================================================
# POSITIVES THAT HAVE ALREADY PARTICIPATED IN METHOD DEVELOPMENT
# ============================================================

dev_positive_cases = {
    # original pilot
    "Cosmos2.5_0016",
    "Cosmos3_seed101_0003",
    "LVP_0004",

    # Expansion-6
    "Cosmos2.5_0058",
    "Cosmos2.5_0060",
    "Cosmos3_seed101_0010",
    "Cosmos3_seed101_0020",
    "Cosmos3_seed101_0036",
    "LVP_0030",
}


# ============================================================
# EXISTING CLEAN CASES ALREADY INSPECTED / USED DURING DEV
# ============================================================

dev_negatives = [
    ("Cosmos2.5", 36),
    ("Cosmos2.5", 57),
    ("Cosmos3_seed101", 39),
    ("Cosmos3_seed101", 49),
    ("LVP", 62),
    ("LVP", 43),
]


# ============================================================
# 20 NEW CLEAN CASES
# PRE-SELECTED BEFORE MODEL SCORING
# ============================================================

heldout_negatives = [

    # seed102: 10
    ("Cosmos3_seed102", 0),
    ("Cosmos3_seed102", 1),
    ("Cosmos3_seed102", 2),
    ("Cosmos3_seed102", 3),
    ("Cosmos3_seed102", 4),
    ("Cosmos3_seed102", 5),
    ("Cosmos3_seed102", 7),
    ("Cosmos3_seed102", 9),
    ("Cosmos3_seed102", 11),
    ("Cosmos3_seed102", 14),

    # seed103: 10
    ("Cosmos3_seed103", 0),
    ("Cosmos3_seed103", 1),
    ("Cosmos3_seed103", 2),
    ("Cosmos3_seed103", 3),
    ("Cosmos3_seed103", 4),
    ("Cosmos3_seed103", 5),
    ("Cosmos3_seed103", 7),
    ("Cosmos3_seed103", 8),
    ("Cosmos3_seed103", 9),
    ("Cosmos3_seed103", 11),
]


# ============================================================
# OBJECT PROMPT INFERENCE
# ============================================================

KNOWN_OBJECTS = [
    "white tape roll",
    "yellow tape roll",
    "brown tape roll",
    "black tape roll",

    "yellow banana",
    "banana",

    "yellow cube",
    "brown cube",
    "white cube",
    "red cube",
    "blue cube",
    "green cube",
    "black cube",

    "white cup",
    "brown cup",
    "yellow cup",
    "red cup",
    "blue cup",
    "green cup",
    "black cup",

    "button",
    "drawer",
]


def infer_object(idx):
    prompt = prompts[idx]["prompt"]
    low = prompt.lower()

    # Prefer explicit known task objects.
    for obj in KNOWN_OBJECTS:
        if obj in low:
            return obj

    patterns = [
        r"picks up the (.+?) from the",
        r"pick up the (.+?) from the",
        r"pushes the (.+?) on the",
        r"presses the (.+?) on the",
        r"grabs the (.+?) from the",
        r"lifts the (.+?) from the",
        r"the (.+?) on the table is picked up",
    ]

    for pat in patterns:
        m = re.search(pat, low)
        if m:
            obj = m.group(1).strip()
            obj = re.sub(r"^(small|large)\s+", "", obj)
            return obj

    raise RuntimeError(
        f"\nCould not infer object for ID {idx:04d}\nPROMPT: {prompt}"
    )


def case_name(dataset, idx):
    return f"{dataset}_{idx:04d}"


def video_path(dataset, idx):

    f = f"{idx:04d}.mp4"

    if dataset == "Cosmos2.5":
        return COS25 / f

    if dataset == "Cosmos3_seed101":
        return COS3 / "seed101" / f

    if dataset == "Cosmos3_seed102":
        return COS3 / "seed102" / f

    if dataset == "Cosmos3_seed103":
        return COS3 / "seed103" / f

    if dataset == "LVP":
        return LVP / f

    raise ValueError(dataset)


rows = []


# ============================================================
# POSITIVE ROWS
# ============================================================

for dataset, ids in positives.items():

    for idx in ids:

        case = case_name(dataset, idx)

        split = (
            "DEV_POS"
            if case in dev_positive_cases
            else "HELDOUT_POS"
        )

        rows.append({
            "case": case,
            "label": 1,
            "eval_split": split,
            "dataset": dataset,
            "object_prompt": infer_object(idx),
            "video_path": str(video_path(dataset, idx)),
            "source_path": str(SOURCE_DIR / f"{idx:04d}.png"),
        })


# ============================================================
# OLD CLEAN NEGATIVES
# ============================================================

for dataset, idx in dev_negatives:

    rows.append({
        "case": case_name(dataset, idx),
        "label": 0,
        "eval_split": "DEV_NEG",
        "dataset": dataset,
        "object_prompt": infer_object(idx),
        "video_path": str(video_path(dataset, idx)),
        "source_path": str(SOURCE_DIR / f"{idx:04d}.png"),
    })


# ============================================================
# NEW CLEAN NEGATIVES
# ============================================================

for dataset, idx in heldout_negatives:

    rows.append({
        "case": case_name(dataset, idx),
        "label": 0,
        "eval_split": "HELDOUT_NEG",
        "dataset": dataset,
        "object_prompt": infer_object(idx),
        "video_path": str(video_path(dataset, idx)),
        "source_path": str(SOURCE_DIR / f"{idx:04d}.png"),
    })


d = pd.DataFrame(rows)

assert d["case"].is_unique


# ============================================================
# FILE EXISTENCE
# ============================================================

bad = []

for _, r in d.iterrows():

    if not Path(r["video_path"]).exists():
        bad.append(("VIDEO", r["case"], r["video_path"]))

    if not Path(r["source_path"]).exists():
        bad.append(("SOURCE", r["case"], r["source_path"]))


print("\n============================================================")
print("IDENTITY EVAL V1")
print("============================================================")

print("\nCounts:")
print(d.groupby(["label","eval_split"]).size())

print("\nTotal:", len(d))
print("Positive:", int((d.label == 1).sum()))
print("Negative:", int((d.label == 0).sum()))

print("\nObject mapping:")
print(
    d[
        ["case","label","eval_split","object_prompt"]
    ].to_string(index=False)
)

if bad:
    print("\nMISSING FILES:")
    for x in bad:
        print(x)
    raise RuntimeError("Missing files detected")


out = "IDENTITY_EVAL_V1.csv"
d.to_csv(out, index=False)

print("\nSAVED:", out)
print("\nAll files verified.")

from pathlib import Path
import numpy as np
import pandas as pd


BASE = Path(
    "/shared/ssd_30T/zhuoyingyang/physact/"
    "sam3_robowm/identity_v1"
)

PAIR_CSV = (
    BASE
    / "dinov2_negative6_frozen"
    / "DINOV2_NEGATIVE6_PAIRS.csv"
)

ASSIGN_CSV = (
    BASE
    / "tracking_diagnostic_negative6"
    / "TRACK_ASSIGNMENTS.csv"
)

TRACK_CSV = (
    BASE
    / "tracking_diagnostic_negative6"
    / "TRACK_SUMMARY.csv"
)

OUT = (
    BASE
    / "persistent_pair_negative6"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEV DIAGNOSTIC SETTING
#
# "persistent" only means the track appears >= 3 sampled frames.
# This is NOT a final tuned threshold.
# ============================================================

MIN_TRACK_OBS = 3


pairs = pd.read_csv(PAIR_CSV)
assign = pd.read_csv(ASSIGN_CSV)
tracks = pd.read_csv(TRACK_CSV)


# ============================================================
# MAP candidate rank -> track
# ============================================================

lookup = assign[
    [
        "case",
        "frame",
        "rank",
        "track_id",
    ]
].copy()

a = lookup.rename(
    columns={
        "rank": "rank_a",
        "track_id": "track_a",
    }
)

b = lookup.rename(
    columns={
        "rank": "rank_b",
        "track_id": "track_b",
    }
)

p = pairs.merge(
    a,
    on=[
        "case",
        "frame",
        "rank_a",
    ],
    how="left",
)

p = p.merge(
    b,
    on=[
        "case",
        "frame",
        "rank_b",
    ],
    how="left",
)


# ============================================================
# TRACK INFORMATION
# ============================================================

ta = tracks[
    [
        "case",
        "track_id",
        "birth_frame",
        "n_observations",
        "lifespan_frames",
    ]
].rename(
    columns={
        "track_id": "track_a",
        "birth_frame": "birth_a",
        "n_observations": "obs_a",
        "lifespan_frames": "life_a",
    }
)

tb = tracks[
    [
        "case",
        "track_id",
        "birth_frame",
        "n_observations",
        "lifespan_frames",
    ]
].rename(
    columns={
        "track_id": "track_b",
        "birth_frame": "birth_b",
        "n_observations": "obs_b",
        "lifespan_frames": "life_b",
    }
)

p = p.merge(
    ta,
    on=["case", "track_a"],
    how="left",
)

p = p.merge(
    tb,
    on=["case", "track_b"],
    how="left",
)


# ============================================================
# INITIAL FRAME PER CASE
# ============================================================

first_frame = (
    assign
    .groupby("case")["frame"]
    .min()
    .to_dict()
)

p["first_frame"] = (
    p["case"]
    .map(first_frame)
)


# ============================================================
# STRUCTURAL ELIGIBILITY
#
# At least one track:
#   1. was not present at initial frame
#   2. eventually persists >= MIN_TRACK_OBS
#
# We use eventual persistence because this is an offline
# video-level physical consistency metric.
# ============================================================

p["post_initial_a"] = (
    p["birth_a"]
    >
    p["first_frame"]
)

p["post_initial_b"] = (
    p["birth_b"]
    >
    p["first_frame"]
)

p["persistent_a"] = (
    p["obs_a"]
    >= MIN_TRACK_OBS
)

p["persistent_b"] = (
    p["obs_b"]
    >= MIN_TRACK_OBS
)

p["new_persistent_a"] = (
    p["post_initial_a"]
    &
    p["persistent_a"]
)

p["new_persistent_b"] = (
    p["post_initial_b"]
    &
    p["persistent_b"]
)

p["has_new_persistent_track"] = (
    p["new_persistent_a"]
    |
    p["new_persistent_b"]
)


# already frozen geometry rule
p["spatially_distinct"] = (
    ~p["nested"]
)


p["eligible"] = (
    p["spatially_distinct"]
    &
    p["has_new_persistent_track"]
)


# ============================================================
# CONTINUOUS TARGET SUPPORT
# ============================================================

p["min_siglip"] = np.minimum(
    p["siglip_source_a"],
    p["siglip_source_b"],
)

p["min_gdino"] = np.minimum(
    p["gdino_a"],
    p["gdino_b"],
)

p["semantic_support"] = np.sqrt(
    np.clip(
        p["min_siglip"],
        0,
        None,
    )
    *
    np.clip(
        p["min_gdino"],
        0,
        None,
    )
)


# No threshold.
p["dup_evidence"] = (
    p["pair_similarity"]
    *
    p["semantic_support"]
)


# Ineligible pair contributes zero.
p["eligible_dup_evidence"] = np.where(
    p["eligible"],
    p["dup_evidence"],
    0.0,
)


# ============================================================
# FRAME SCORE
# ============================================================

best_rows = []

for (case, frame), g in p.groupby(
    ["case", "frame"]
):

    g = g.sort_values(
        "eligible_dup_evidence",
        ascending=False,
    )

    best = g.iloc[0]

    best_rows.append({
        "case": case,
        "frame": int(frame),

        "score":
            float(
                best[
                    "eligible_dup_evidence"
                ]
            ),

        "pair_similarity":
            float(
                best[
                    "pair_similarity"
                ]
            ),

        "semantic_support":
            float(
                best[
                    "semantic_support"
                ]
            ),

        "min_siglip":
            float(
                best[
                    "min_siglip"
                ]
            ),

        "min_gdino":
            float(
                best[
                    "min_gdino"
                ]
            ),

        "rank_a":
            int(
                best["rank_a"]
            ),

        "rank_b":
            int(
                best["rank_b"]
            ),

        "track_a":
            best["track_a"],

        "track_b":
            best["track_b"],

        "birth_a":
            int(
                best["birth_a"]
            ),

        "birth_b":
            int(
                best["birth_b"]
            ),

        "obs_a":
            int(
                best["obs_a"]
            ),

        "obs_b":
            int(
                best["obs_b"]
            ),

        "eligible":
            bool(
                best["eligible"]
            ),
    })


frames = pd.DataFrame(
    best_rows
)


# ============================================================
# SAVE
# ============================================================

p.to_csv(
    OUT / "PAIR_DIAGNOSTIC.csv",
    index=False,
)

frames.to_csv(
    OUT / "FRAME_DIAGNOSTIC.csv",
    index=False,
)


# ============================================================
# PRINT CASE TRAJECTORIES
# ============================================================

print("=" * 115)
print("PERSISTENT NEW-TRACK PAIR DIAGNOSTIC")
print("=" * 115)

for case, g in frames.groupby(
    "case"
):

    print()
    print("=" * 115)
    print(case)
    print("=" * 115)

    top = (
        g.sort_values(
            "score",
            ascending=False,
        )
        .head(12)
    )

    print(
        top[
            [
                "frame",
                "score",
                "pair_similarity",
                "semantic_support",
                "min_siglip",
                "min_gdino",
                "rank_a",
                "rank_b",
                "track_a",
                "track_b",
                "birth_a",
                "birth_b",
                "obs_a",
                "obs_b",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# HUMAN-AUDIT FRAMES
# ============================================================

audit = {
    "Cosmos2.5_0058":
        [12,15,18,24,27,33,45,87],

    "Cosmos2.5_0060":
        [0,6,12,18,27,33,45,90],

    "Cosmos3_seed101_0010":
        [0,33,60,72,90,132,156],

    "Cosmos3_seed101_0020":
        [120,126,129,132,138,141,144,174,186],

    "Cosmos3_seed101_0036":
        [0,30,93,123,129,132,135,171,183],

    "LVP_0030":
        [18,24,39,42,45,48],
}


print()
print("=" * 115)
print("AUDIT FRAMES")
print("=" * 115)


for case, fs in audit.items():

    print()
    print(case)

    z = frames[
        (frames["case"] == case)
        &
        (frames["frame"].isin(fs))
    ].copy()

    print(
        z[
            [
                "frame",
                "score",
                "pair_similarity",
                "semantic_support",
                "min_siglip",
                "min_gdino",
                "track_a",
                "track_b",
                "birth_a",
                "birth_b",
                "obs_a",
                "obs_b",
            ]
        ].to_string(
            index=False
        )
    )


print()
print("DONE")
print(
    "PAIR:",
    OUT / "PAIR_DIAGNOSTIC.csv"
)
print(
    "FRAME:",
    OUT / "FRAME_DIAGNOSTIC.csv"
)

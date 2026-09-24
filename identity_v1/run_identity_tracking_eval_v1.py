from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from transformers import AutoImageProcessor, Dinov2Model


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    "/shared/ssd_30T/zhuoyingyang/physact"
)

BASE = (
    ROOT / "sam3_robowm/identity_v1"
)

MANIFEST = (
    BASE / "IDENTITY_EVAL_V1.csv"
)

CANDIDATES = (
    BASE
    / "sam_identity_eval_v1"
    / "SAM_IDENTITY_EVAL_V1_CANDIDATES.csv"
)

OUT = (
    BASE / "tracking_identity_eval_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

ASSIGN_CSV = (
    OUT / "TRACK_ASSIGNMENTS.csv"
)

TRACK_CSV = (
    OUT / "TRACK_SUMMARY.csv"
)

BIRTH_CSV = (
    OUT / "TRACK_BIRTH_DIAGNOSTICS.csv"
)


# ============================================================
# DINOv2
# ============================================================

MODEL_NAME = "facebook/dinov2-base"

CACHE = (
    "/shared/ssd_30T/zhuoyingyang/models/hf_cache"
)


# ============================================================
# DIAGNOSTIC TRACKING SETTINGS
#
# IMPORTANT:
# These are NOT final duplication thresholds.
# They are only permissive continuity settings for examining
# track birth / persistence.
# ============================================================

FRAME_STEP = 3

# Allow a track to disappear for one sampled frame
# and reconnect at the next one.
MAX_FRAME_GAP = 6

# Matching cost:
#   alpha * appearance_difference
# + (1-alpha) * center_motion
APPEARANCE_WEIGHT = 0.75

# Permissive continuity gates.
MIN_TRACK_DINO = 0.25
MAX_CENTER_DISTANCE = 0.35
MAX_MATCH_COST = 0.65

BATCH_SIZE = 32


# ============================================================
# HELPERS
# ============================================================

def center_of(row):
    return np.array([
        (float(row["x1"]) + float(row["x2"])) / 2.0,
        (float(row["y1"]) + float(row["y2"])) / 2.0,
    ], dtype=np.float32)


def normalized_center_distance(
    center_a,
    center_b,
    W,
    H,
):
    diag = np.sqrt(
        float(W) ** 2
        + float(H) ** 2
    )

    if diag <= 0:
        return 0.0

    return float(
        np.linalg.norm(
            center_a - center_b
        ) / diag
    )


# ============================================================
# LOAD DATA
# ============================================================

manifest = pd.read_csv(
    MANIFEST
)

d = pd.read_csv(
    CANDIDATES
)

d = (
    d.sort_values(
        ["case", "frame", "rank"]
    )
    .reset_index(drop=True)
)

d["row_id"] = np.arange(
    len(d)
)


# ============================================================
# VIDEO DIMENSIONS
# ============================================================

video_info = {}

for _, r in manifest.iterrows():

    case = r["case"]
    video = r["video_path"]

    cap = cv2.VideoCapture(
        str(video)
    )

    W = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    H = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    cap.release()

    video_info[case] = {
        "W": W,
        "H": H,
    }


# ============================================================
# LOAD DINOv2
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 100)
print("IDENTITY TRACKING DIAGNOSTIC")
print("=" * 100)

print("device:", device)
print("candidate rows:", len(d))
print("cases:", d["case"].nunique())

processor = (
    AutoImageProcessor
    .from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE,
        local_files_only=True,
    )
)

model = (
    Dinov2Model
    .from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE,
        local_files_only=True,
    )
    .to(device)
    .eval()
)

print("DINOv2 ready.")


@torch.no_grad()
def embed_images(images):

    x = processor(
        images=images,
        return_tensors="pt",
    )

    out = model(
        pixel_values=
            x["pixel_values"].to(device)
    )

    f = (
        out.last_hidden_state[:, 0]
    )

    f = (
        f
        / (
            f.norm(
                dim=-1,
                keepdim=True
            )
            + 1e-8
        )
    )

    return (
        f.detach()
        .cpu()
        .numpy()
    )


# ============================================================
# EMBED ALL SAM CROPS
# ============================================================

print()
print("Embedding SAM crops...")

features = np.zeros(
    (len(d), 768),
    dtype=np.float32,
)

for start in range(
    0,
    len(d),
    BATCH_SIZE,
):

    stop = min(
        len(d),
        start + BATCH_SIZE
    )

    imgs = []

    for p in d.loc[
        start:stop-1,
        "crop_path"
    ]:
        imgs.append(
            Image.open(
                p
            ).convert("RGB")
        )

    f = embed_images(
        imgs
    )

    features[
        start:stop
    ] = f

    print(
        f"{stop}/{len(d)}"
    )

print("Embeddings ready.")


# ============================================================
# TRACKING
# ============================================================

assignment_rows = []
track_summary_rows = []
birth_rows = []


for case in d["case"].unique():

    print()
    print("=" * 100)
    print(case)
    print("=" * 100)

    z = (
        d[d["case"] == case]
        .copy()
    )

    W = video_info[case]["W"]
    H = video_info[case]["H"]

    frames = sorted(
        z["frame"]
        .astype(int)
        .unique()
    )

    # tracks[track_id]
    tracks = {}

    next_track_id = 0


    # --------------------------------------------------------
    # PROCESS FRAME BY FRAME
    # --------------------------------------------------------

    for frame in frames:

        g = (
            z[z["frame"] == frame]
            .sort_values("rank")
            .reset_index(drop=True)
        )

        current = []

        for _, r in g.iterrows():

            rid = int(
                r["row_id"]
            )

            current.append({
                "row": r,
                "row_id": rid,
                "feat": features[rid],
                "center": center_of(r),
            })


        # ----------------------------------------------------
        # Tracks still eligible for continuity
        # ----------------------------------------------------

        active_ids = []

        for tid, tr in tracks.items():

            gap = (
                int(frame)
                - int(tr["last_frame"])
            )

            if gap <= MAX_FRAME_GAP:
                active_ids.append(
                    tid
                )


        # ----------------------------------------------------
        # Build all possible track-candidate edges
        # ----------------------------------------------------

        edges = []

        for tid in active_ids:

            tr = tracks[tid]

            for ci, c in enumerate(
                current
            ):

                sim = float(
                    np.dot(
                        tr["last_feat"],
                        c["feat"]
                    )
                )

                dist = (
                    normalized_center_distance(
                        tr["last_center"],
                        c["center"],
                        W,
                        H,
                    )
                )

                cost = (
                    APPEARANCE_WEIGHT
                    * (1.0 - sim)
                    +
                    (1.0 - APPEARANCE_WEIGHT)
                    * dist
                )

                # Permissive continuity gate
                eligible = (
                    sim >= MIN_TRACK_DINO
                    and
                    dist <= MAX_CENTER_DISTANCE
                    and
                    cost <= MAX_MATCH_COST
                )

                if eligible:

                    edges.append({
                        "track_id": tid,
                        "candidate_index": ci,
                        "sim": sim,
                        "dist": dist,
                        "cost": cost,
                    })


        # ----------------------------------------------------
        # Greedy one-to-one assignment.
        #
        # Only <=3 candidates per frame, so this diagnostic
        # matching is sufficient and transparent.
        # ----------------------------------------------------

        edges = sorted(
            edges,
            key=lambda x:
                x["cost"]
        )

        used_tracks = set()
        used_candidates = set()

        matched = {}

        for e in edges:

            tid = e["track_id"]
            ci = e["candidate_index"]

            if tid in used_tracks:
                continue

            if ci in used_candidates:
                continue

            used_tracks.add(tid)
            used_candidates.add(ci)

            matched[ci] = e


        # ----------------------------------------------------
        # UPDATE MATCHED TRACKS FIRST
        # ----------------------------------------------------

        visible_existing_tracks = []

        for ci, e in matched.items():

            c = current[ci]
            tid = e["track_id"]

            tr = tracks[tid]

            tr["last_frame"] = int(
                frame
            )

            tr["last_feat"] = (
                c["feat"]
            )

            tr["last_center"] = (
                c["center"]
            )

            tr["history"].append({
                "frame": int(frame),
                "row_id": c["row_id"],
                "rank":
                    int(
                        c["row"]["rank"]
                    ),
            })

            visible_existing_tracks.append(
                tid
            )

            assignment_rows.append({
                "case": case,
                "frame": int(frame),
                "rank":
                    int(
                        c["row"]["rank"]
                    ),
                "track_id": tid,
                "is_birth": False,

                "match_dino":
                    e["sim"],

                "center_distance":
                    e["dist"],

                "match_cost":
                    e["cost"],

                "source_similarity":
                    float(
                        c["row"][
                            "source_similarity"
                        ]
                    ),

                "gdino_score":
                    float(
                        c["row"][
                            "gdino_score"
                        ]
                    ),

                "crop_path":
                    c["row"][
                        "crop_path"
                    ],
            })


        # ----------------------------------------------------
        # CREATE NEW TRACKS FOR UNMATCHED CANDIDATES
        # ----------------------------------------------------

        for ci, c in enumerate(
            current
        ):

            if ci in used_candidates:
                continue


            # -----------------------------------------------
            # Compare NEW candidate to tracks that are
            # currently visible in the same frame.
            #
            # This tells us:
            # "does this newly born instance look like an
            # already-existing visible instance?"
            # -----------------------------------------------

            best_existing_tid = ""
            best_existing_sim = np.nan

            if visible_existing_tracks:

                comparisons = []

                for tid in visible_existing_tracks:

                    sim = float(
                        np.dot(
                            tracks[tid][
                                "last_feat"
                            ],
                            c["feat"]
                        )
                    )

                    comparisons.append(
                        (sim, tid)
                    )

                comparisons.sort(
                    reverse=True
                )

                best_existing_sim = (
                    comparisons[0][0]
                )

                best_existing_tid = (
                    comparisons[0][1]
                )


            tid = (
                f"T{next_track_id:02d}"
            )

            next_track_id += 1


            tracks[tid] = {
                "track_id": tid,

                "birth_frame":
                    int(frame),

                "last_frame":
                    int(frame),

                "last_feat":
                    c["feat"],

                "last_center":
                    c["center"],

                "birth_source_similarity":
                    float(
                        c["row"][
                            "source_similarity"
                        ]
                    ),

                "birth_gdino_score":
                    float(
                        c["row"][
                            "gdino_score"
                        ]
                    ),

                "birth_existing_track":
                    best_existing_tid,

                "birth_existing_similarity":
                    best_existing_sim,

                "history": [{
                    "frame":
                        int(frame),

                    "row_id":
                        c["row_id"],

                    "rank":
                        int(
                            c["row"]["rank"]
                        ),
                }],
            }


            assignment_rows.append({
                "case": case,
                "frame": int(frame),
                "rank":
                    int(
                        c["row"]["rank"]
                    ),

                "track_id": tid,
                "is_birth": True,

                "match_dino":
                    np.nan,

                "center_distance":
                    np.nan,

                "match_cost":
                    np.nan,

                "source_similarity":
                    float(
                        c["row"][
                            "source_similarity"
                        ]
                    ),

                "gdino_score":
                    float(
                        c["row"][
                            "gdino_score"
                        ]
                    ),

                "crop_path":
                    c["row"][
                        "crop_path"
                    ],
            })


            birth_rows.append({
                "case": case,
                "track_id": tid,

                "birth_frame":
                    int(frame),

                "birth_rank":
                    int(
                        c["row"]["rank"]
                    ),

                "source_similarity":
                    float(
                        c["row"][
                            "source_similarity"
                        ]
                    ),

                "gdino_score":
                    float(
                        c["row"][
                            "gdino_score"
                        ]
                    ),

                "most_similar_existing_track":
                    best_existing_tid,

                "similarity_to_existing":
                    best_existing_sim,
            })


        # ----------------------------------------------------
        # PRINT CURRENT FRAME
        # ----------------------------------------------------

        frame_assign = [
            r for r in assignment_rows
            if (
                r["case"] == case
                and
                r["frame"] == frame
            )
        ]

        text = []

        for r in frame_assign:

            flag = (
                "NEW"
                if r["is_birth"]
                else "match"
            )

            text.append(
                f"R{r['rank']}->{r['track_id']}({flag})"
            )

        print(
            f"frame={int(frame):03d}  "
            + "  ".join(text)
        )


    # --------------------------------------------------------
    # CASE TRACK SUMMARY
    # --------------------------------------------------------

    print()
    print("TRACK SUMMARY")

    for tid, tr in tracks.items():

        hist = tr["history"]

        frames_hist = [
            h["frame"]
            for h in hist
        ]

        row_ids = [
            h["row_id"]
            for h in hist
        ]

        zz = d.iloc[
            row_ids
        ]

        n_obs = len(hist)

        birth = int(
            tr["birth_frame"]
        )

        last = int(
            tr["last_frame"]
        )

        lifespan = (
            last - birth
        )

        source_median = float(
            zz["source_similarity"]
            .median()
        )

        gdino_median = float(
            zz["gdino_score"]
            .median()
        )

        track_summary_rows.append({
            "case": case,
            "track_id": tid,

            "birth_frame":
                birth,

            "last_frame":
                last,

            "n_observations":
                n_obs,

            "lifespan_frames":
                lifespan,

            "median_source_similarity":
                source_median,

            "median_gdino_score":
                gdino_median,

            "birth_existing_track":
                tr[
                    "birth_existing_track"
                ],

            "birth_existing_similarity":
                tr[
                    "birth_existing_similarity"
                ],
        })

        sim = (
            tr[
                "birth_existing_similarity"
            ]
        )

        sim_text = (
            "nan"
            if pd.isna(sim)
            else f"{sim:.3f}"
        )

        print(
            f"{tid}: "
            f"birth={birth:3d} "
            f"last={last:3d} "
            f"obs={n_obs:2d} "
            f"life={lifespan:3d} "
            f"src_med={source_median:.3f} "
            f"birth_sim={sim_text} "
            f"to={tr['birth_existing_track']}"
        )


# ============================================================
# SAVE
# ============================================================

assign_df = pd.DataFrame(
    assignment_rows
)

track_df = pd.DataFrame(
    track_summary_rows
)

birth_df = pd.DataFrame(
    birth_rows
)


assign_df.to_csv(
    ASSIGN_CSV,
    index=False
)

track_df.to_csv(
    TRACK_CSV,
    index=False
)


# Add persistence information to birth table
if len(birth_df):

    birth_df = birth_df.merge(
        track_df[
            [
                "case",
                "track_id",
                "last_frame",
                "n_observations",
                "lifespan_frames",
                "median_source_similarity",
                "median_gdino_score",
            ]
        ],
        on=[
            "case",
            "track_id"
        ],
        how="left",
    )

    birth_df.to_csv(
        BIRTH_CSV,
        index=False
    )


# ============================================================
# PRINT POST-INITIAL TRACK BIRTHS
# ============================================================

print()
print("=" * 100)
print("POST-INITIAL TRACK BIRTHS")
print("=" * 100)

for case in d["case"].unique():

    first_frame = int(
        d[
            d["case"] == case
        ]["frame"].min()
    )

    q = track_df[
        (track_df["case"] == case)
        &
        (
            track_df["birth_frame"]
            > first_frame
        )
    ].copy()

    q = q.sort_values(
        [
            "n_observations",
            "birth_existing_similarity",
        ],
        ascending=[
            False,
            False,
        ]
    )

    print()
    print(case)

    if not len(q):
        print(
            "No post-initial births."
        )
        continue

    print(
        q[
            [
                "track_id",
                "birth_frame",
                "last_frame",
                "n_observations",
                "lifespan_frames",
                "median_source_similarity",
                "median_gdino_score",
                "birth_existing_track",
                "birth_existing_similarity",
            ]
        ].to_string(
            index=False
        )
    )


print()
print("=" * 100)
print("DONE")
print("=" * 100)

print(
    "ASSIGNMENTS:",
    ASSIGN_CSV
)

print(
    "TRACK SUMMARY:",
    TRACK_CSV
)

print(
    "BIRTHS:",
    BIRTH_CSV
)

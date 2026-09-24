#!/usr/bin/env bash

set -uo pipefail

ROOT="/shared/ssd_30T/zhuoyingyang/physact/sam3_robowm/identity_v1"
LOGDIR="$ROOT/overnight_identity_eval_logs"

mkdir -p "$LOGDIR"

cd "$ROOT"

source ~/miniconda3/etc/profile.d/conda.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/shared/ssd_30T/zhuoyingyang/physact/sam3_robowm/sam3:${PYTHONPATH:-}"

echo "======================================================================"
echo "IDENTITY OVERNIGHT PIPELINE"
echo "Started: $(date)"
echo "======================================================================"

# ----------------------------------------------------------------------
# Safety: make sure all required inputs/scripts exist
# ----------------------------------------------------------------------

REQUIRED=(
    "IDENTITY_EVAL_V1.csv"
    "run_sam_identity_eval_v1.py"
    "run_dinov2_identity_eval_v1.py"
    "run_identity_tracking_eval_v1.py"
    "run_persistent_pair_eval_v1.py"
    "evaluate_identity_auc_v1.py"
    "siglip_identity_eval_v1/SIGLIP_IDENTITY_EVAL_V1_CANDIDATES.csv"
)

for f in "${REQUIRED[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing required file:"
        echo "$f"
        exit 1
    fi
done

echo
echo "Preflight OK."
echo


# ======================================================================
# GPU FUNCTIONS
# ======================================================================

pick_gpu () {

    local NEED="$1"

    nvidia-smi \
        --query-gpu=index,memory.free \
        --format=csv,noheader,nounits \
    | awk -F',' -v need="$NEED" '
        {
            gsub(/[[:space:]]/, "", $1)
            gsub(/[[:space:]]/, "", $2)

            if (($2 + 0) >= need && ($2 + 0) > maxfree) {
                maxfree = $2 + 0
                gpu = $1
            }
        }
        END {
            if (gpu != "")
                print gpu
        }
    '
}


wait_for_gpu () {

    local NEED="$1"

    while true; do

        GPU=$(pick_gpu "$NEED")

        if [[ -n "${GPU:-}" ]]; then

            FREE=$(nvidia-smi \
                --query-gpu=index,memory.free \
                --format=csv,noheader,nounits \
                | awk -F',' -v g="$GPU" '
                    {
                        gsub(/[[:space:]]/, "", $1)
                        gsub(/[[:space:]]/, "", $2)
                        if ($1 == g) print $2
                    }
                ')

            echo "[$(date)] GPU $GPU available: ${FREE} MiB free" >&2
            echo "$GPU"
            return
        fi

        echo "[$(date)] No GPU with >= ${NEED} MiB free. Checking again in 60 sec..." >&2

        nvidia-smi \
            --query-gpu=index,memory.used,memory.free \
            --format=csv,noheader >&2

        sleep 60
    done
}


run_gpu_stage () {

    local NAME="$1"
    local LOG="$2"
    local NEED="$3"
    local OUTDIR="$4"

    shift 4

    while true; do

        GPU=$(wait_for_gpu "$NEED")

        echo
        echo "======================================================================"
        echo "$NAME"
        echo "GPU = $GPU"
        echo "Started = $(date)"
        echo "======================================================================"

        # Clear only an incomplete previous attempt.
        rm -rf "$OUTDIR"

        set +e

        CUDA_VISIBLE_DEVICES="$GPU" \
        "$@" 2>&1 | tee "$LOG"

        STATUS=${PIPESTATUS[0]}

        set -e

        if [[ "$STATUS" -eq 0 ]]; then
            echo
            echo "$NAME completed successfully: $(date)"
            return 0
        fi

        echo
        echo "$NAME exited with status $STATUS"

        if grep -qiE \
            "CUDA out of memory|OutOfMemoryError|out of memory" \
            "$LOG"; then

            echo "OOM detected."
            echo "Will wait for another sufficiently free GPU and retry."
            sleep 120

        else
            echo
            echo "NON-OOM ERROR."
            echo "Stopping overnight pipeline."
            echo "Check:"
            echo "$LOG"
            exit "$STATUS"
        fi
    done
}


# ======================================================================
# STEP 1 — SAM3
# ======================================================================

SAM_FILE="sam_identity_eval_v1/SAM_IDENTITY_EVAL_V1_CANDIDATES.csv"

if [[ -s "$SAM_FILE" ]]; then

    echo "SAM3 output already exists — skipping."

else

    # SAM used ~6 GB in our previous run.
    # Require 12 GB free to give comfortable headroom.
    run_gpu_stage \
        "SAM3" \
        "$LOGDIR/01_sam3.log" \
        12000 \
        "sam_identity_eval_v1" \
        conda run --no-capture-output -n sam3 \
        python run_sam_identity_eval_v1.py
fi


if [[ ! -s "$SAM_FILE" ]]; then
    echo "ERROR: SAM completed but candidate CSV is missing."
    exit 1
fi

echo
python - <<'PY'
import pandas as pd

f = "sam_identity_eval_v1/SAM_IDENTITY_EVAL_V1_CANDIDATES.csv"
d = pd.read_csv(f)

print("SAM candidate rows :", len(d))
print("SAM cases present  :", d["case"].nunique())

expected = set(pd.read_csv("IDENTITY_EVAL_V1.csv")["case"])
got = set(d["case"])

print("Cases with no SAM candidate rows:")
print(sorted(expected - got))
PY


# ======================================================================
# STEP 2 — DINOV2
# ======================================================================

DINO_FILE="dinov2_identity_eval_v1/DINOV2_IDENTITY_EVAL_V1_PAIRS.csv"

if [[ -s "$DINO_FILE" ]]; then

    echo
    echo "DINOv2 output already exists — skipping."

else

    run_gpu_stage \
        "DINOv2" \
        "$LOGDIR/02_dinov2.log" \
        10000 \
        "dinov2_identity_eval_v1" \
        conda run --no-capture-output -n rbench_vlm \
        python run_dinov2_identity_eval_v1.py
fi


# ======================================================================
# STEP 3 — TRACKING
# ======================================================================

TRACK_FILE="tracking_identity_eval_v1/TRACK_ASSIGNMENTS.csv"

if [[ -s "$TRACK_FILE" ]]; then

    echo
    echo "Tracking output already exists — skipping."

else

    run_gpu_stage \
        "TRACKING" \
        "$LOGDIR/03_tracking.log" \
        10000 \
        "tracking_identity_eval_v1" \
        conda run --no-capture-output -n rbench_vlm \
        python run_identity_tracking_eval_v1.py
fi


# ======================================================================
# STEP 4 — PERSISTENT PAIR
# ======================================================================

echo
echo "======================================================================"
echo "PERSISTENT PAIR"
echo "Started: $(date)"
echo "======================================================================"

PAIR_FILE="persistent_pair_identity_eval_v1/PAIR_DIAGNOSTIC.csv"

if [[ -s "$PAIR_FILE" ]]; then

    echo "Persistent-pair output already exists — skipping."

else

    rm -rf persistent_pair_identity_eval_v1

    conda run --no-capture-output -n rbench_vlm \
        python run_persistent_pair_eval_v1.py \
        2>&1 | tee "$LOGDIR/04_persistent_pair.log"

    STATUS=${PIPESTATUS[0]}

    if [[ "$STATUS" -ne 0 ]]; then
        echo "Persistent-pair stage failed."
        exit "$STATUS"
    fi
fi


# ======================================================================
# STEP 5 — FINAL SPATIAL-INDEPENDENCE SCORE + AUROC
# ======================================================================

echo
echo "======================================================================"
echo "FINAL AUROC"
echo "Started: $(date)"
echo "======================================================================"

conda run --no-capture-output -n rbench_vlm \
    python evaluate_identity_auc_v1.py \
    2>&1 | tee "$LOGDIR/05_auc.log"

STATUS=${PIPESTATUS[0]}

if [[ "$STATUS" -ne 0 ]]; then
    echo "AUROC evaluation failed."
    exit "$STATUS"
fi


# ======================================================================
# DONE
# ======================================================================

echo
echo "======================================================================"
echo "ALL DONE"
echo "Finished: $(date)"
echo "======================================================================"

echo
echo "FINAL SCORE OUTPUT:"
echo

tail -100 "$LOGDIR/05_auc.log"

{
    echo "IDENTITY EVAL V1 COMPLETE"
    echo "Finished: $(date)"
    echo
    tail -100 "$LOGDIR/05_auc.log"
} > IDENTITY_EVAL_OVERNIGHT_DONE.txt

echo
echo "Saved:"
echo "$ROOT/IDENTITY_EVAL_OVERNIGHT_DONE.txt"
echo "$ROOT/IDENTITY_EVAL_V1_VIDEO_SCORES.csv"

import os
import subprocess
import time
from pathlib import Path

import pandas as pd


ROOT = Path(
    "/shared/ssd_30T/zhuoyingyang/physact/"
    "sam3_robowm/identity_v1"
)

MANIFEST = ROOT / "IDENTITY_EVAL_V1.csv"
BASE_SCRIPT = ROOT / "run_sam_identity_eval_v1.py"

WORK = ROOT / "sam_identity_eval_casewise"
MANIFEST_DIR = WORK / "manifests"
SCRIPT_DIR = WORK / "scripts"
LOG_DIR = WORK / "logs"

MASTER_DIR = ROOT / "sam_identity_eval_v1"
MASTER_CSV = (
    MASTER_DIR /
    "SAM_IDENTITY_EVAL_V1_CANDIDATES.csv"
)

for p in [
    WORK,
    MANIFEST_DIR,
    SCRIPT_DIR,
    LOG_DIR,
    MASTER_DIR,
]:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# GPU helpers
# ============================================================

def gpu_status():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ]

    out = subprocess.check_output(
        cmd,
        text=True,
    )

    rows = []

    for line in out.strip().splitlines():
        gpu, free = line.split(",")
        rows.append(
            (
                int(gpu.strip()),
                int(free.strip()),
            )
        )

    # Most free first
    rows.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return rows


# ============================================================
# Merge all completed cases into the normal expected CSV
# ============================================================

def merge_completed():

    dfs = []
    completed = []

    for case_dir in sorted(WORK.glob("case_*")):

        done = case_dir / "DONE"

        csv = (
            case_dir /
            "SAM_CASE_CANDIDATES.csv"
        )

        if done.exists() and csv.exists():

            try:
                d = pd.read_csv(csv)
            except Exception as e:
                print(
                    f"WARNING: cannot read {csv}: {e}"
                )
                continue

            dfs.append(d)
            completed.append(
                case_dir.name[len("case_"):]
            )

    if dfs:

        out = pd.concat(
            dfs,
            ignore_index=True
        )

        out.to_csv(
            MASTER_CSV,
            index=False
        )

        print(
            f"\nCHECKPOINT MERGED: "
            f"{len(completed)} cases, "
            f"{len(out)} candidate rows"
        )

    else:

        print(
            "\nNo completed cases yet."
        )

    return completed


# ============================================================
# Data
# ============================================================

manifest = pd.read_csv(MANIFEST)

print("=" * 100)
print("SAM3 IDENTITY EVAL — CASE-WISE FP32")
print("=" * 100)

print("Total cases:", len(manifest))
print("Precision: ORIGINAL FP32")
print(
    "Each case runs in a fresh process "
    "and is checkpointed immediately."
)


# ============================================================
# Main
# ============================================================

for index, row in manifest.iterrows():

    case = row["case"]

    safe_case = case.replace("/", "_")

    case_dir = (
        WORK /
        f"case_{safe_case}"
    )

    case_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    done_file = case_dir / "DONE"

    case_csv = (
        case_dir /
        "SAM_CASE_CANDIDATES.csv"
    )

    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------

    if (
        done_file.exists()
        and case_csv.exists()
    ):

        print(
            f"\n[{index+1:02d}/{len(manifest)}] "
            f"{case}: ALREADY DONE -> skip"
        )

        continue


    # --------------------------------------------------------
    # One-row manifest
    # --------------------------------------------------------

    case_manifest = (
        MANIFEST_DIR /
        f"{safe_case}.csv"
    )

    pd.DataFrame(
        [row]
    ).to_csv(
        case_manifest,
        index=False
    )


    # --------------------------------------------------------
    # Create one-case copy of ORIGINAL SAM script
    # --------------------------------------------------------

    s = BASE_SCRIPT.read_text()

    # Important:
    # only change I/O paths.
    # No model parameters or precision changes.
    s = s.replace(
        "IDENTITY_EVAL_V1.csv",
        str(case_manifest),
    )

    s = s.replace(
        "sam_identity_eval_v1",
        str(case_dir),
    )

    s = s.replace(
        "SAM_IDENTITY_EVAL_V1_CANDIDATES.csv",
        "SAM_CASE_CANDIDATES.csv",
    )

    case_script = (
        SCRIPT_DIR /
        f"run_{safe_case}.py"
    )

    case_script.write_text(s)


    # --------------------------------------------------------
    # Try the three currently freest GPUs.
    #
    # No waiting loop.
    # If one OOMs, immediately try another GPU.
    # --------------------------------------------------------

    success = False

    candidates = gpu_status()[:3]

    print()
    print("=" * 100)
    print(
        f"[{index+1:02d}/{len(manifest)}] "
        f"{case}"
    )
    print("=" * 100)

    print(
        "GPU candidates:",
        candidates
    )


    for attempt, (gpu, free_mb) in enumerate(
        candidates,
        start=1
    ):

        print(
            f"\nAttempt {attempt}/{len(candidates)}"
            f"  GPU={gpu}"
            f"  free={free_mb} MiB"
        )

        log = (
            LOG_DIR /
            f"{safe_case}_attempt{attempt}"
            f"_gpu{gpu}.log"
        )

        env = os.environ.copy()

        env[
            "CUDA_VISIBLE_DEVICES"
        ] = str(gpu)

        env[
            "PYTORCH_CUDA_ALLOC_CONF"
        ] = "expandable_segments:True"

        old_pp = env.get(
            "PYTHONPATH",
            ""
        )

        sam_path = (
            "/shared/ssd_30T/zhuoyingyang/"
            "physact/sam3_robowm/sam3"
        )

        env["PYTHONPATH"] = (
            sam_path
            + (":" + old_pp if old_pp else "")
        )


        cmd = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "sam3",
            "python",
            str(case_script),
        ]


        with open(log, "w") as f:

            p = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in p.stdout:
                print(line, end="")
                f.write(line)
                f.flush()

            rc = p.wait()


        if (
            rc == 0
            and case_csv.exists()
        ):

            done_file.write_text(
                "SUCCESS\n"
            )

            print(
                f"\nSUCCESS: {case}"
            )

            success = True

            # Save global checkpoint immediately
            merge_completed()

            break


        # ----------------------------------------------------
        # Failure diagnosis
        # ----------------------------------------------------

        text = log.read_text(
            errors="ignore"
        )

        if (
            "OutOfMemoryError" in text
            or "CUDA out of memory" in text
        ):

            print(
                f"\nOOM on GPU {gpu}. "
                "Trying another GPU."
            )

        else:

            print(
                f"\nFAILED on GPU {gpu} "
                f"(return code {rc})."
            )

            print(
                "See:",
                log
            )


        # Remove partial output for ONLY this case.
        # Completed previous cases are untouched.
        if case_csv.exists():
            case_csv.unlink()

        time.sleep(3)


    if not success:

        print()
        print(
            f"*** {case} NOT COMPLETED ***"
        )

        print(
            "Moving to the next case. "
            "Run this same script again later; "
            "completed cases will be skipped."
        )


# ============================================================
# Final merge / summary
# ============================================================

completed = merge_completed()

expected = set(
    manifest["case"]
)

completed_set = set(
    completed
)

missing = sorted(
    expected - completed_set
)


print()
print("=" * 100)
print("FINAL CASE-WISE SAM SUMMARY")
print("=" * 100)

print(
    "Completed:",
    len(completed_set),
    "/",
    len(expected)
)

print(
    "Missing:",
    len(missing)
)

for case in missing:
    print(
        "  PENDING:",
        case
    )


if len(missing) == 0:

    print()
    print(
        "ALL SAM CASES COMPLETE."
    )

    print(
        "Master CSV:"
    )

    print(
        MASTER_CSV
    )

else:

    print()
    print(
        "Some cases remain pending."
    )

    print(
        "Simply rerun:"
    )

    print(
        "python "
        "run_sam_identity_eval_casewise.py"
    )

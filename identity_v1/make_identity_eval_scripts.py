from pathlib import Path


def make(src, dst, replacements):

    s = Path(src).read_text()

    for a,b in replacements:
        s = s.replace(a,b)

    Path(dst).write_text(s)

    print("CREATED:", dst)


# ============================================================
# GDINO
# ============================================================

make(
    "run_gdino_expansion6.py",
    "run_gdino_identity_eval_v1.py",
    [
        ("EXPANSION6_FROZEN.csv", "IDENTITY_EVAL_V1.csv"),
        ("gdino_expansion6_frozen", "gdino_identity_eval_v1"),
        ("GDINO_EXPANSION6_COUNTS.csv",
         "GDINO_IDENTITY_EVAL_V1_COUNTS.csv"),
        ("GDINO_EXPANSION6_PROPOSALS.csv",
         "GDINO_IDENTITY_EVAL_V1_PROPOSALS.csv"),
    ]
)


# ============================================================
# SIGLIP
# ============================================================

make(
    "run_siglip_source_match_expansion6.py",
    "run_siglip_identity_eval_v1.py",
    [
        ("EXPANSION6_FROZEN.csv", "IDENTITY_EVAL_V1.csv"),
        ("gdino_expansion6_frozen", "gdino_identity_eval_v1"),
        ("GDINO_EXPANSION6_PROPOSALS.csv",
         "GDINO_IDENTITY_EVAL_V1_PROPOSALS.csv"),
        ("siglip_source_match_expansion6",
         "siglip_identity_eval_v1"),
        ("SIGLIP_EXPANSION6_CANDIDATES.csv",
         "SIGLIP_IDENTITY_EVAL_V1_CANDIDATES.csv"),
        ("SIGLIP_EXPANSION6_FRAME_SUMMARY.csv",
         "SIGLIP_IDENTITY_EVAL_V1_FRAME_SUMMARY.csv"),
    ]
)


# ============================================================
# SAM3
# ============================================================

make(
    "run_sam_masked_expansion6.py",
    "run_sam_identity_eval_v1.py",
    [
        ("EXPANSION6_FROZEN.csv", "IDENTITY_EVAL_V1.csv"),
        ("siglip_source_match_expansion6",
         "siglip_identity_eval_v1"),
        ("SIGLIP_EXPANSION6_CANDIDATES.csv",
         "SIGLIP_IDENTITY_EVAL_V1_CANDIDATES.csv"),
        ("sam_masked_expansion6",
         "sam_identity_eval_v1"),
        ("SAM_EXPANSION6_CANDIDATES.csv",
         "SAM_IDENTITY_EVAL_V1_CANDIDATES.csv"),
    ]
)


# ============================================================
# DINOV2
# ============================================================

make(
    "run_dinov2_expansion6.py",
    "run_dinov2_identity_eval_v1.py",
    [
        ("sam_masked_expansion6",
         "sam_identity_eval_v1"),
        ("SAM_EXPANSION6_CANDIDATES.csv",
         "SAM_IDENTITY_EVAL_V1_CANDIDATES.csv"),
        ("dinov2_expansion6_frozen",
         "dinov2_identity_eval_v1"),
        ("DINOV2_EXPANSION6_PAIRS.csv",
         "DINOV2_IDENTITY_EVAL_V1_PAIRS.csv"),
        ("DINOV2_EXPANSION6_FRAMES.csv",
         "DINOV2_IDENTITY_EVAL_V1_FRAMES.csv"),
        ("DINOV2_EXPANSION6_CASE_SUMMARY.csv",
         "DINOV2_IDENTITY_EVAL_V1_CASE_SUMMARY.csv"),
    ]
)


# ============================================================
# TRACKING
# use the already-fixed negative6 version
# ============================================================

make(
    "run_identity_tracking_negative6.py",
    "run_identity_tracking_eval_v1.py",
    [
        ("NEGATIVE6_FROZEN.csv",
         "IDENTITY_EVAL_V1.csv"),
        ("sam_masked_negative6",
         "sam_identity_eval_v1"),
        ("SAM_NEGATIVE6_CANDIDATES.csv",
         "SAM_IDENTITY_EVAL_V1_CANDIDATES.csv"),
        ("tracking_diagnostic_negative6",
         "tracking_identity_eval_v1"),
    ]
)


# ============================================================
# PERSISTENT PAIR
# ============================================================

make(
    "run_persistent_pair_negative6.py",
    "run_persistent_pair_eval_v1.py",
    [
        ("dinov2_negative6_frozen",
         "dinov2_identity_eval_v1"),
        ("DINOV2_NEGATIVE6_PAIRS.csv",
         "DINOV2_IDENTITY_EVAL_V1_PAIRS.csv"),
        ("tracking_diagnostic_negative6",
         "tracking_identity_eval_v1"),
        ("persistent_pair_negative6",
         "persistent_pair_identity_eval_v1"),
    ]
)

print("\nDONE")

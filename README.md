# Duplication Identity Analysis

Evaluation of object duplication and identity inconsistency in generated robot-manipulation videos.

## Final method

Final scoring script:

`identity_v1/final_size_spatial_ablation.py`

Final video scores:

`identity_v1/FINAL_SIZE_SPATIAL_VIDEO_SCORES.csv`

Final pair scores:

`identity_v1/FINAL_SIZE_SPATIAL_PAIR_SCORES.csv`

Pipeline:

GroundingDINO proposals
-> source-conditioned SigLIP
-> SAM3 segmentation
-> DINOv2 pair similarity
-> semantic support
-> size consistency
-> spatial independence

## Final results

Strict reaudited evaluation:

- 54 videos
- 24 duplication positives
- 30 negatives

Primary score:

`size_spatial_peak`

AUROC: **0.8528**

Strict numerical variant:

`strict_size_spatial_peak`

AUROC: **0.8625**

## Current limitation

This is a duplication detector, not yet a complete object-persistence system.

The next stage expands consistency to:

- duplication
- disappearance
- spontaneous appearance
- splitting / fragmentation
- identity persistence
- deformation

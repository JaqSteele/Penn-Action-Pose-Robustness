# Frame extraction, verification and RTMPose inference

This document covers Batch C: frame extraction, the D-01 frame checks, and the RTMPose-M inference pipeline. The protocol (`docs/RESEARCH_PROTOCOL.md`, v0.12) is authoritative.

**Status:**
- training-partition frames are extracted and verified;
- the five-video smoke test is complete;
- **full training-partition inference has not been run** and needs separate approval;
- test-partition frames have **not** been extracted or opened.

## Where things are stored (all outside the repository)

```
C:\datasets\penn-action\
  frames\<video_id>\000001.jpg ...          extracted frames (training partition only)
  manifests\
    labels-sha256.txt, archive-sha256.txt   integrity manifests (labels, archive)
    frames-train-sha256.txt                 SHA-256 of every extracted training frame
    extraction-train.json                   extraction summary
    d01-frames-train.json                   D-01 frame-check report (per video)
  poses\rtmpose_m\
    smoke\run1\, run2\                      two full GPU runs of the 5 smoke-test videos
    smoke\gpu_sample\, cpu_sample\          every 10th frame on each device (CPU cross-check)
    smoke\overlays\                         15 overlay images
    smoke\report.json                       smoke-test checks and preliminary E1
```

## Commands (from the repository root)

| Step | Environment | Command |
|---|---|---|
| Extract training frames | main | `.venv\Scripts\python.exe -B scripts\extract_frames.py --partition train` |
| D-01 frame checks | main | `.venv\Scripts\python.exe -B scripts\verify_frames.py --partition train` |
| Smoke test, run 1 | RTMPose | `C:\venvs\penn-action-rtmpose\Scripts\python.exe -B scripts\run_rtmpose.py --smoke --out-dir C:\datasets\penn-action\poses\rtmpose_m\smoke\run1` |
| Smoke test, run 2 (determinism) | RTMPose | same command with `...\smoke\run2` |
| CPU and GPU samples | RTMPose | add `--device cpu --frame-step 10 --sample-only` (and `--device cuda:0 ...`) with `...\smoke\cpu_sample` / `gpu_sample` |
| Smoke checks and preliminary E1 | main | `.venv\Scripts\python.exe -B scripts\check_smoke.py` |
| Overlays | main | `.venv\Scripts\python.exe -B scripts\make_overlays.py` |
| Tests | main | `.venv\Scripts\python.exe -B -m unittest discover -s tests -v` |

**Safety:** every script refuses to overwrite existing outputs. Test-partition frames can only be extracted or verified with an explicit `--allow-test`, which has not been used. `run_rtmpose.py` accepts training-partition videos only.

## Inference settings (protocol D-03, D-04, D-05, D-34a, v0.12)

**Model and checks before loading:**
- RTMPose-M, 256 × 192, MMPose 1.3.2, loaded with `init_model(config, checkpoint, device)` (no `cfg_options`).
- The config and checkpoint SHA-256 values are verified first.
- The model's `flip_test` setting, its COCO-17 keypoint names and FP32 weights are asserted.

**Per frame:**
- `inference_topdown(model, <frame path>, bboxes=<one box>, bbox_format='xyxy')`.
- The box comes from the D-05 rules: the frame's own valid box; otherwise the nearest valid box in the video (the earlier one on ties); otherwise the full image.

**Numerical settings:** cuDNN benchmark off; cuDNN deterministic on; TF32 off for both cuDNN and matmul.

**Validity (D-34a):** a keypoint is invalid if its score is ≤ 0, its score is not finite, or a coordinate is not finite. Invalid keypoints are flagged and never imputed here; their handling (D-34) comes later.

**Per-video output** (`<id>.npz` plus `<id>.json`):

| Array | Shape | Meaning |
|---|---|---|
| `frame_index` | (N,) | 1 … N |
| `image_size` | (2,) | actual [W, H] |
| `bbox_annotation` | (N, 4) | aligned annotation box (NaN where no row) |
| `bbox_used` | (N, 4) | box given to the estimator |
| `bbox_replaced` | (N,) | frame's own box was invalid |
| `keypoints17`, `scores17`, `valid17` | (N, 17, 2), (N, 17), (N, 17) | raw RTMPose output and D-34a flags |
| `keypoints12`, `scores12`, `valid12` | (N, 12, 2), (N, 12), (N, 12) | canonical COCO 5–16 selection |

The JSON records the settings, torch flags, hashes, library versions, device, Git commit (and whether the working tree had changes), timings and fallback counts.

## Smoke-test selection

The five videos are evenly spaced positions in the sorted fitting list, `floor(k × 1069 / 4)` for k = 0 … 4: **0002, 0516, 1101, 1746, 2325**. The selection depends only on the stored split, not on pose quality.

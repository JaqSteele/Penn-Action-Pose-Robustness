# Implementation notes

This document explains how the code implements the research protocol (`docs/RESEARCH_PROTOCOL.md`, v0.11). The protocol is authoritative: if the two ever disagree, the protocol wins and the code must be fixed.

**Status:**
- **Batch B:** foundations.
- **Batch C:** frame extraction and checks, plus the RTMPose pipeline and its five-video smoke test (see `docs/INFERENCE.md`).
- **Not yet done:** full training-partition inference, recognition models and experiments.

## Layout

| Path | Contents |
|---|---|
| `src/penn_pose/labels.py` | Loading and validating annotation files; the test-partition lock |
| `src/penn_pose/splits.py` | The deterministic fitting/validation split (D-02) |
| `src/penn_pose/mapping.py` | The 12-joint primary representation and left/right correction (D-08, D-09, D-11) |
| `src/penn_pose/boxes.py` | Box validity, fallbacks and per-video centre/scale (D-05, D-12) |
| `src/penn_pose/gt_preprocess.py` | Corner-placeholder repair (D-07b), and the ground-truth pipeline |
| `src/penn_pose/normalise.py` | Per-video normalisation and its inverse (D-12) |
| `src/penn_pose/integrity.py` | Annotation-file integrity checks against the SHA-256 manifest |
| `src/penn_pose/frames.py` | Frame naming, listing and D-01 frame checks (Batch C) |
| `src/penn_pose/pose_io.py` | D-34a validity flags; per-video estimator output format (Batch C) |
| `src/penn_pose/inference_config.py` | RTMPose settings, hashes and the smoke-test selection (Batch C) |
| `src/penn_pose/e1.py` | Measurement E1 diagnostics for invalid estimates (Batch C) |
| `scripts/` | Extraction, verification, inference, smoke checks and overlays (see `docs/INFERENCE.md`) |
| `configs/splits/fitting.txt`, `validation.txt` | The stored split lists (identifiers only) |
| `tests/` | Standard-library `unittest` tests |

## Data location

- The dataset stays **outside the repository**. The default location is `C:\datasets\penn-action`; set the environment variable `PENN_ACTION_ROOT` to use another location.
- Expected contents:
  - `labels/*.mat`: the 2,326 annotation files;
  - `manifests/labels-sha256.txt`: the SHA-256 manifest of those files;
  - `manifests/archive-sha256.txt`: the archive's SHA-256.
- No dataset file is ever written or modified by this code.

## Running the tests

From the repository root, with the main environment:

```powershell
& ".\.venv\Scripts\python.exe" -B -m unittest discover -s tests -v
```

Tests that need the dataset are skipped automatically if `labels/` is not found.

## Test-partition protection (protocol Section 4.3)

- **Partition first.** `labels.read_partition_flag` reads **only** a file's `train` variable. Every other loader checks this flag first.
- **Blocked by default.** `labels.load_video` raises `TestPartitionAccessError` for a test-partition video unless the caller passes `allow_test=True`. Listing the `test` or `all` partitions needs the same opt-in.
- **Reserved opt-in.** `allow_test=True` is reserved for the final, frozen evaluation (Section 9.5.7). No code in this batch uses it.
- **What touches test files at all:** reading the `train` flag, to know which videos are in the test partition; and hashing file bytes, for the integrity check. No other field of a test-partition file is read.

## Joint order (protocol v0.12, Section 6.2)

The canonical order of the 12 primary joints is **COCO keypoints 5–16**, named by the subject's **anatomical** side (established in protocol v0.12):

`left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle`

- **Ground truth:** each joint comes from the **opposite-labelled** Penn Action column (D-09). For example, `left_shoulder` comes from column 2, labelled "Right shoulder (3)". The full table is in `mapping.py`.
- **Estimates:** RTMPose keypoints 5–16 map in the same order, without reordering, after checking the estimator's keypoint names.
- **Source mapping versus canonical order:** the protocol's Section 6.2 table, in Penn Action column order, defines only the source mapping. The canonical order is COCO 5–16.
- **Name check:** `tests/test_keypoint_names.py` checks the code's COCO names against the names recorded from the checkpoint in `configs/pose/rtmpose_m.yaml`.

## Implementation choices within the protocol

- **Image width:** a `dimensions` width of 481 or 482 is treated as a 480-pixel frame, as the D-05 amendment states. Other widths are used as recorded.
- **Strict loading:** joint rows, `nframes` and the frame count in `dimensions` must agree, or loading fails with `LabelFormatError`. Box rows may differ in number from frames; they are handled by D-05 row alignment.
- **D-05 tie rule:** when two valid frames are equally near, the earlier frame is used.
- **D-07b rule 4:** "partner track placeholder-free after rule 3" is implemented as "the partner is not itself an entirely placeholder track", because rule 3 removes every placeholder from partly placeholder tracks.
- **Split files and line endings:** the repository has `core.autocrlf=true`, so a checkout could convert the files to CRLF line endings. `splits.load_split` therefore parses the file strictly, then re-serialises the identifiers in the canonical form (LF-separated, final LF, ASCII) before checking the recorded SHA-256. The stored files are written with LF line endings, and their raw bytes also match the recorded hashes.

## What the tests establish

- **Split:** 1,258 / 1,070 / 188 videos; the per-class counts; both recorded SHA-256 values; determinism; and that the stored files match.
- **D-05:** the validity rules, row alignment, the nearest-valid tie rule and the full-image fallback. Video 1516 gets centre (240, 175.5) and scale √(480² + 351²) ≈ 594.64. It is the only training video without a valid box.
- **D-07b (training partition):**
  - 14 partly placeholder tracks;
  - 601 partner copies and 494 video-centre fallbacks (1,095 in total);
  - 78,166 placeholder points;
  - no placeholder remaining after repair;
  - no visible joint ever a placeholder.
- **Mapping:** the full Penn → primary table, every pair swapped, partners symmetric, and the COCO mapping checked against the keypoint names. The COCO names are also checked against the checkpoint record in the YAML.
- **D-12:** the normalisation formula, the same scale for x and y, and a round trip within 1e-9 px on every training video.
- **Integrity:** all 2,326 label files match the manifest. Test-partition loading and listing are blocked without opt-in.

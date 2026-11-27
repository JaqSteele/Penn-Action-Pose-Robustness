# Recognition pipeline: recogniser inputs (Batch D1)

This document covers Batch D1: building the Release 1.0 recogniser inputs for the **training partition** from ground-truth poses and RTMPose-M estimates. The protocol (`docs/RESEARCH_PROTOCOL.md`, v0.13) is authoritative; if this document and the protocol disagree, the protocol wins.

**Scope.** D1 covers preprocessing and dataset construction only: no model, no training, no model selection and no metrics. It does not access the test partition.

## Pipeline

| Step | Ground truth (`"gt"`) | Estimates (`"est"`) | Protocol |
|---|---|---|---|
| 1. 12 primary joints | Penn columns, left/right swapped | RTMPose COCO keypoints 5–16 | D-08, D-09, D-11, v0.12 |
| 2. Unreliable coordinates | Corner-placeholder repair (`gt_preprocess`) | Invalid-keypoint imputation, steps 1–4 (`est_preprocess`), in pixels | D-07b; D-34a, D-34b |
| 3. Normalisation | `(p − c_v) / s_v` from the video's valid GT boxes | the same `c_v` and `s_v`, obtained through the locked annotation loader | D-12, D-05 |
| 4. Resampling | 64 frames at `t_k = k(N − 1)/63` (`temporal`) | the same | D-20 |

Normalisation and resampling are identical for both sources.

**Estimated poses use estimator outputs and image size only.**
- They never use ground-truth joint coordinates, visibility or placeholder logic.
- The only ground-truth information they use is the per-video median box centre and diagonal (the documented D-05 exception to D-07a rule 4).
- A test confirms this: replacing a video's ground-truth joints with NaN and inverting its visibility leaves the estimated inputs bit-identical.

**Imputation (D-34a step 1, D-34b steps 2–4).**
1. Linear interpolation in time within the same joint, holding the nearest valid value at the start and end of the video.
2. A track never valid in the video copies its partner's track (after step 1), if the partner has at least one valid estimate.
3. If the partner is also never valid, use the per-video mean of the **original** valid primary estimates.
4. If no primary keypoint is valid anywhere in the video, use the image centre `(W/2, H/2)` from the pose output's actual `image_size`. The video is flagged.

**Cross-checks for every estimated video.**
- The pose output's frame count must equal the annotation's frame count.
- Its `image_size` must equal the annotation's image size (481/482 corrected to 480).
- Its stored `bbox_annotation` must equal the annotation boxes after D-05 row alignment.

Any mismatch raises `EstimateConsistencyError`.

## Input format

- `X`: `(n, 24, 64)` float32. Processing is in float64; the result is cast to float32 at the end.
- **Channels are interleaved:** channel `2j` is x and channel `2j + 1` is y of primary joint `j`, in COCO order 5–16. `dataset.CHANNEL_NAMES` gives `left_shoulder_x, left_shoulder_y, right_shoulder_x, …, right_ankle_y`.
- `y`: int64 class index. **Class map:** the dataset's own `action` strings, sorted with Python `sorted()` (index 0 is `baseball_pitch`, index 14 is `tennis_serve`). It is recorded in `dataset.CLASS_NAMES`, checked against the training labels, and persisted in `class_map.json`.
- `ids` (`<U4`, ascending), `subset` (`fitting` or `validation`, from the hash-checked split lists) and `n_frames` (original lengths).

## Safeguards

- Every identifier's partition flag is read first. A test-partition identifier raises `TestPartitionAccessError` before anything else is read.
- Identifiers must be unique and present in the stored D-02 lists.
- The ground-truth and estimated sets must agree in identifiers, order, labels, subsets and original frame counts. All values must be finite.
- The build refuses to run if the output folder (or `<out>.incomplete`) exists. Files are written to `<out>.incomplete` and renamed only after every check passes.
- Before building, label files and pose outputs are checked against their SHA-256 manifests.
- If the imputation does not reproduce E1, the build stops without writing anything.

## Cache (outside the repository)

```
C:\datasets\penn-action\inputs\release1\
  gt_train.npz, est_train.npz   X, y, ids, subset, n_frames (+ per-video step counts for est)
  class_map.json                class names and indices
  build_report.json             git state, source hashes, integrity checks, imputation summary, statistics
  inputs-sha256.txt             SHA-256 manifest of the four files above
```

## Commands (main environment, repository root)

| Step | Command |
|---|---|
| Build, then verify | `.venv\Scripts\python.exe -B scripts\build_inputs.py` |
| Verify an existing build | `.venv\Scripts\python.exe -B scripts\build_inputs.py --verify` |
| Tests | `.venv\Scripts\python.exe -B -m unittest discover -s tests -v` |

Verification checks:
- the manifest hashes and the exact file set;
- for the default cache, the manifest's SHA-256 and entries against the repository record `configs/inputs/release1-cache.json`;
- the label and pose-output files against their SHA-256 manifests, rechecked independently of the build;
- the array names in each `.npz` file, the array formats and the class map;
- that the identifiers are the 1,258 training videos and the subsets match the split lists;
- that an independent in-memory rebuild is bit-identical to the cache, including the per-video imputation-step counts in `est_train.npz`.

## Durable record of the accepted cache

`configs/inputs/release1-cache.json` records the accepted D1 cache: the SHA-256 of `inputs-sha256.txt` (`2ab40e9263a903def05a1975ab1be659e40ad9532fc0ab8dff4d7564b2ae8e55`) and of each of the four files it lists. The cache was built at commit `45399c8` with the D1 files not yet committed. Its `build_report.json` therefore records `dirty: true`, and the SHA-256 of every source file used. That report predates the hardening pass, so it has no SciPy version, test-file hashes or time zone. Builds made after the hardening pass record all three.

## Results of the D1 build (training partition)

- 1,258 videos (1,070 fitting / 188 validation) for each source; `X` has shape `(1258, 24, 64)`.
- **Ground-truth repair:** 14 partly-placeholder tracks, 601 partner copies, 494 `c_v` fallbacks and 78,166 placeholder points. These match protocol Section 5.5.3.
- **Imputation:** 20 observations by step 1, in videos 0359, 0362, 0373, 0462, 0710, 1516 and 1905; 0 by steps 2–4. This reproduces E1: 20 in the fitting subset and 0 in the validation subset.

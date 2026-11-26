# Pose Estimation Setup: RTMPose-M

**Status:** software environment installed and smoke-tested. **Not validated on Penn Action.**
**Recorded:** 2026-11-25
**Related:** [`environments/rtmpose/`](../environments/rtmpose/README.md) (environment), [`configs/pose/rtmpose_m.yaml`](../configs/pose/rtmpose_m.yaml) (settings), [`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md) (methodology)

> **Scope of this document.** It records *which* estimator is installed, *how* it is configured, and *that* the software works. The smoke tests described here verify software functionality only. They do **not** measure pose-estimation accuracy on Penn Action, and they must not be cited as evidence of research accuracy. The estimator choice and settings are not yet formally frozen in the research protocol (decisions D-03, D-04 and D-05 remain open there).

## How to read the evidence labels

| Label | Meaning |
|---|---|
| **[SOURCE]** | Read from the official config or the installed MMPose 1.3.2 source code |
| **[EXECUTED]** | Confirmed by running code in the RTMPose environment |
| **[REPORTED]** | From the official model zoo or paper, not re-measured here |
| **[PROPOSED]** | A recommended setting that still needs formal approval |

---

## 1. Model

RTMPose-M is a top-down 2D human pose estimator. It receives an image crop around one person and predicts 17 COCO body keypoints.

| Property | Value | Evidence |
|---|---|---|
| Backbone | CSPNeXt (P5), `deepen_factor=0.67`, `widen_factor=0.75`, channel attention. Defined in MMDetection (`_scope_='mmdet'`). | [SOURCE] |
| Head | RTMCCHead: 7×7 conv, then a gated attention unit (`hidden_dims=256`, `s=128`, `expansion_factor=2`, SiLU, dropout 0, drop-path 0), then SimCC classifiers | [SOURCE] |
| Output representation | SimCC: separate x and y classification vectors, 2 bins per input pixel (`simcc_split_ratio=2.0`), no DARK refinement | [SOURCE] |
| Input size | 192 × 256 (width × height) | [SOURCE] |
| Keypoints | 17, COCO order (see section 6) | [EXECUTED] |
| Parameters / FLOPs | 13.59 M / 1.93 GFLOPs | [REPORTED] RTMPose README |
| COCO val2017 AP | 74.6 (with a 56.4-AP person detector and flip testing) | [REPORTED] MMPose model zoo |

**Training history** (relevant for interpreting results and licensing). The model-zoo label "Training Data: COCO" describes only the final stage:

1. **CSPNeXt backbone:** ImageNet classification pretraining (from RTMDet).
2. **Pose pretraining:** UDP heatmap pretraining, 210 epochs, on COCO + AI Challenger. *This is inferred from the backbone checkpoint name* `cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth`; the public pretraining configs in MMPose are COCO-only.
3. **Final model:** SimCC fine-tuning, 420 epochs, on COCO train2017 (this config).

Judging by their filenames and the RTMPose documentation, none of the official RTMPose COCO, AIC+COCO or Body7 checkpoints is free of AI Challenger data. This is inferred, not stated by OpenMMLab. Other checkpoint families, such as the Human-Art models, were not checked.

## 2. Configuration

| Item | Value |
|---|---|
| MMPose version | 1.3.2 (PyPI) |
| Config | `configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py` |
| Config SHA-256 | `f27da50a81c3d3b5c426262118b9a030d15c9dbddccda7100b8cc0bcd07f6fff` |
| Base config | `configs/_base_/default_runtime.py`, SHA-256 `bb7df18270b1ca337192faae7a416b2dea23b45e798e832caab5781f41189d06` |

During setup, a separately obtained copy of these two files was used, stored outside the repository in `C:\venvs\penn-action-rtmpose-setup\mmpose-v1.3.2\configs\`. Where that copy was originally downloaded from was **not independently verified**. What *is* verified is that it is **byte-identical** to the copy bundled inside the installed `mmpose` 1.3.2 package (`mmpose/.mim/configs/...`), with the same SHA-256 values as above [EXECUTED]. Neither file is copied into this repository.

## 3. Checkpoint identity

| Item | Value |
|---|---|
| File | `rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth` |
| URL | `https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth` |
| Size | 54,535,816 bytes |
| MD5 | `b498ff40dc64220592b3ec9736cfa26e` (computed from the local file [EXECUTED]) |
| SHA-256 | `929f00f2d84764aa0297decf13a84556642e43cbf5fca8cfb16e7326d92dfd97` |
| Contents | `state_dict` (362 tensors) and `meta` (including `dataset_meta`); no optimizer state [EXECUTED] |
| Local location | `C:\venvs\penn-action-rtmpose-setup\checkpoints\` (outside the repository) |

**Download-time integrity check [REPORTED by the researcher].** At download time, the researcher reports checking the file against the download server's `Content-MD5` header and the expected file size. That comparison with the server was **not independently re-verified**. What was independently confirmed [EXECUTED] is the local file's size, MD5 and SHA-256, as listed above. The SHA-256 is the project's reference identifier from now on.

**Note on the filename tag.** The 8-character tag `d8dd5ca4` in the filename does **not** match the start of this file's SHA-256 (`929f00f2…`). The tag should therefore not be treated as a checksum for this file. The link between this local file and the official download rests on the researcher-reported MD5 and size check against the server.

The checkpoint is **not** stored in or distributed through this repository. `*.pth` and `checkpoints/*` are excluded by `.gitignore`.

## 4. Current default inference settings (verified, not yet frozen)

These are the values the official MMPose pipeline uses **today**. Verifying them establishes what the estimator currently does. It is **not** a decision to keep them: the proposed settings to freeze are listed separately in `configs/pose/rtmpose_m.yaml` (`proposed_freeze`) and await formal approval.

| Setting | Current value | Evidence |
|---|---|---|
| Flip testing | `flip_test=True`. The crop is also run mirrored; the mirrored prediction is flipped back, left/right joints are swapped, and the two are averaged. | [SOURCE][EXECUTED] |
| Box padding | ×1.25 (default of `GetBBoxCenterScale`; the config passes no value) | [SOURCE][EXECUTED] |
| Aspect ratio | The padded box is enlarged (never cut) to 3:4 to match 192 × 256 | [SOURCE] |
| Crop and resize | Affine warp of the box region to 192 × 256 with bilinear interpolation (`cv2.INTER_LINEAR`), no rotation. No border arguments are passed, so areas outside the image use OpenCV's default constant (black) border. | [SOURCE] |
| Colour | Images are loaded as **BGR** (OpenCV). The model converts BGR to RGB internally (`bgr_to_rgb=True`), then normalises with mean `[123.675, 116.28, 103.53]` and std `[58.395, 57.12, 57.375]`. | [SOURCE][EXECUTED] |
| Box format | `bbox_format='xyxy'` (left, top, right, bottom, pixels) | [SOURCE][EXECUTED] |
| Decoding | Argmax per axis. Score = the smaller of the two axis maxima (raw, not a probability; a score of 1.022 was observed). Scores ≤ 0 set the location to −1 *before* conversion to image coordinates. | [SOURCE][EXECUTED] |
| Coordinate resolution | One SimCC bin is 0.5 input px. In image pixels this depends on the box size: 0.903 px for the recorded test box. | [SOURCE][EXECUTED] |
| Backbone pretrained initialisation | Disabled: `init_model` sets `backbone.init_cfg = None` when `cfg_options` is not passed, and never calls `init_weights()`. No download occurred. | [SOURCE][EXECUTED] |
| SyncBN | Converted to standard batch norm by `init_model` | [SOURCE][EXECUTED] |
| Random operations | None in the inference pipeline or model (eval mode, no dropout). Repeated runs were bit-identical on GPU and on CPU. | [SOURCE][EXECUTED] |
| PyTorch numerical defaults | `cudnn.benchmark=False`, `cudnn.deterministic=False`, `cudnn.allow_tf32=True`; MMPose does not change them | [EXECUTED] |

### 4.1 Sensitivity checks (illustrative, n = 1)

To understand which settings matter, each setting was changed **one at a time** from the defaults above, and the resulting keypoints were compared with the default output [EXECUTED].

- **Image:** the official MMPose skier test photograph (640 × 425 px). It is not stored in the repository.
- **Box:** the recorded test box `[270, 35, 485, 405]`, `xyxy`.
- **Design:** the same model instance, image and box, with exactly one setting changed per comparison. Inference is deterministic, so each difference is caused by the changed setting alone.
- **Sample size:** one image, one box, one person.
- **Metrics,** computed over all 17 keypoints and in image pixels:
  - *largest keypoint distance*: the largest straight-line (Euclidean) distance any keypoint moved;
  - *mean keypoint distance*: the average of those distances;
  - *largest coordinate change*: the largest change in a single x or y value.

| Change from default | Largest keypoint distance (px) | Mean keypoint distance (px) | Largest coordinate change (px) | Keypoints that moved |
|---|---|---|---|---|
| Flip testing off | 2.02 | 0.95 | 1.81 | 13 / 17 |
| Padding 1.0 instead of 1.25 | 3.58 | 1.42 | 2.53 | 17 / 17 |
| Padding 1.5 instead of 1.25 | 4.17 | 1.37 | 4.16 | 17 / 17 |
| RGB array passed instead of BGR (no error raised) | 2.55 | 0.94 | 1.81 | 12 / 17 |
| `xywh` box passed as `xyxy` | 81.51 | 19.09 | 76.35 | 17 / 17 |

**About the last row.** Passing `[270, 35, 215, 370]` (the `xywh` form) as `xyxy` makes the right edge (215) lie to the **left** of the left edge (270), so the model receives an **inverted, invalid box**. The row shows that a box-format error can be catastrophic rather than slight. Project code must therefore convert Penn Action boxes explicitly and **validate** them (right > left, bottom > top) before inference.

**Limitations.** These figures describe one image and one box and are **not statistical estimates**:
- They change with the box. During the investigation, a slightly different box (`[270, 40, 510, 400]`) gave different values, and even a different ranking of the two padding changes.
- Keypoint positions change in whole SimCC bins (about 0.9 image px here), so small differences are discrete steps rather than continuous shifts.
- They say nothing about accuracy, on COCO or Penn Action. They only show that each setting can change the output, which is why each must be fixed and recorded.

### 4.2 Numerical settings, CPU vs GPU, and throughput

These checks used the investigation box `[270, 40, 510, 400]` on the same image (n = 1) [EXECUTED]:

- Switching TF32 off, or setting `cudnn.deterministic=True`, changed no keypoint (0.0000 px).
- CPU and GPU produced identical keypoints; scores differed by at most 0.0001.
- **Throughput:** about 35 ms per crop on the RTX 3060 Ti (flip on, including Python overhead; peak memory 196 MiB), and about 102 ms per crop on the CPU (6 threads).

A single image cannot show that these settings never matter: over many frames, a keypoint lying on a bin boundary could still change by one bin. This is why `configs/pose/rtmpose_m.yaml` proposes setting them explicitly.

## 5. Smoke tests

### 5.1 Tests completed during setup [REPORTED by the researcher]

1. MMPose inference API imported successfully.
2. Configuration loaded successfully.
3. Checkpoint loaded onto CUDA successfully.
4. Synthetic image inference returned 17 keypoints.
5. Real-image inference returned 17 keypoints and keypoint scores. The image was an official MMPose test photograph of a skier, and the explicit box was **`[270, 35, 485, 405]` with `bbox_format='xyxy'`**.
6. An annotated skeleton visualisation was generated.

The test photograph and its overlay are kept outside the repository and are **not redistributed**. The photograph is a third-party (COCO/Flickr) image with its own licence.

### 5.2 Re-execution checks (2026-11-25) [EXECUTED]

| Check | Result |
|---|---|
| Versions match the lock file; `mmcv.ops` compiled extensions import | Pass |
| `init_model` on `cuda:0`; backbone `init_cfg` is `None`; no file downloaded (empty `TORCH_HOME`) | Pass |
| Real image, box `[270, 35, 485, 405]`, `xyxy` | Output `(1, 17, 2)` keypoints and `(1, 17)` scores; all finite; all inside the padded crop; scores 0.658–1.022 |
| Same input run twice | Bit-identical |
| Agreement with the saved overlay from test 5 | Median 0.99 px to the drawn keypoints (13 of 17 within 2 px), consistent with the recorded box |
| Synthetic black image with a box | 17 keypoints returned; low scores (0.03–0.13) |
| Keypoint order and flip pairs | COCO-17 order; `flip_indices = [0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15]` |

**What these tests show:** the environment, configuration, checkpoint and inference pipeline work, and accept a supplied ground-truth-style box.

**What they do not show:** accuracy on Penn Action, or anything about the estimator's behaviour on Penn Action frames. The test photograph comes from COCO, the same domain the model was trained on.

## 6. Keypoints and the research joint mapping

The model outputs the **17 COCO keypoints** in this order [EXECUTED]: 0 nose, 1 left eye, 2 right eye, 3 left ear, 4 right ear, 5 left shoulder, 6 right shoulder, 7 left elbow, 8 right elbow, 9 left wrist, 10 right wrist, 11 left hip, 12 right hip, 13 left knee, 14 right knee, 15 left ankle, 16 right ankle.

The primary research analysis uses **12 joints** (shoulders, elbows, wrists, hips, knees, ankles), as defined in `RESEARCH_PROTOCOL.md` §6. That 17-to-12 mapping is a **separate component that is not yet implemented**.

**Left/right convention (D-09): resolved in protocol v0.4.** Penn Action columns labelled LEFT hold the subject's anatomical RIGHT, so every left/right pair is **swapped** when mapping Penn Action to these COCO keypoints. The evidence and limitations are in `RESEARCH_PROTOCOL.md` §6.4. The decision rests on Penn Action annotations and frames, not on RTMPose output. RTMPose's own side convention is *assumed* to be anatomical. That assumption is consistent with one test photograph, where the subject faces the camera and the model's `left_wrist` appeared on the image's right side, but it has not been verified beyond that.

The Penn Action joint ordering (D-11) remains **not yet formally verified**.

## 7. Licensing (unresolved)

| Component | Status |
|---|---|
| MMPose, MMCV, MMEngine, MMDetection code | Apache License 2.0 |
| RTMPose pretrained checkpoint | **No official licence statement found.** OpenMMLab's licence files cover the code. Maintainers have not answered open GitHub issues asking which terms apply to RTMPose checkpoints (e.g. [open-mmlab/mmpose#3282](https://github.com/open-mmlab/mmpose/issues/3282), [#3277](https://github.com/open-mmlab/mmpose/issues/3277)). One informal maintainer reply ([#3062](https://github.com/open-mmlab/mmpose/issues/3062)) concerns a different (Body7) checkpoint. |
| Training data inherited by the checkpoint | ImageNet, COCO and AI Challenger. Published terms for AI Challenger were not found. |

**Handling in this project:** the checkpoint is used for non-commercial academic research only. It is identified by URL and SHA-256 rather than redistributed, and no weights or derived pose data are published. This remains open as protocol decision D-30.

## 8. References

- Jiang, T. et al. (2023). *RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose.* arXiv:2303.07399.
- MMPose: <https://github.com/open-mmlab/mmpose> (v1.3.2); RTMPose project page: <https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose>
- MMCV prebuilt wheel index used: <https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html>

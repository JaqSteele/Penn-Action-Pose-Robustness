# Research Protocol (Draft)

**Project:** Penn Action: Pose Estimation Robustness in Human Action Recognition
**Document status:** Draft v0.8 for review. Not final.
**Date of this revision:** 2026-11-26 (v0.1 drafted 2026-11-18)
**Project status at time of writing:** Environment and repository set up. The Penn Action dataset has been downloaded and is stored locally, outside the repository. Its annotation files and a small sample of training frames have been inspected. No poses have been estimated and no experiments have been run. This document contains **no experimental results**.

### Revision history

| Version | Date | Summary |
|---|---|---|
| v0.1 | 2026-11-18 | First draft. |
| v0.2 | 2026-11-20 | Targeted revision following independent methodological review. Changes: a primary model-selection strategy (C1 validation, hyperparameters frozen across conditions); a 12-joint primary representation with the head excluded; corruption residuals restricted to the fitting subset; confidence scores excluded from recogniser inputs; visibility handling clarified; repeated C4 corruption draws favoured; paired bootstrap procedure specified; H6 framed as exploratory; PCK definition clarified. The research question, scope and conditions C1–C5 are unchanged. |
| v0.3 | 2026-11-20 | Resolved D-33. After selection on C1 validation and freezing of all settings, final models are reinitialised and trained from scratch on the full official training partition (1,258 videos), with the frozen epoch count and no validation-based early stopping. The corruption model remains derived from the fitting subset only. Sections 4.1, 4.4, 4.5, 7.1–7.4, 8.1, 9.2, 9.3, 11.2, 12 and 14 updated accordingly. No other methodological decisions changed. **v0.3 amendments after internal review (same date):** §4.3 design-stage pose-error analysis restricted to the fitting subset (previously "training partition"); §4.4 Stage B adds that data-fitted preprocessing is refitted on the full official training partition and never on the test partition, with methods still open under D-12 (later Stage B items renumbered); §11.2 Stage A log label corrected to "fitting + validation subsets"; §12 limitation widened from the epoch count to all frozen hyperparameters. No decisions resolved or changed. |
| v0.4 | 2026-11-26 | Resolved D-09 (left/right convention). Inspection of Penn Action training annotations and frames showed that columns labelled LEFT hold the subject's anatomical RIGHT, and vice versa. The Penn Action → COCO mapping therefore swaps every left/right pair (Sections 6.2, 6.4, 14). The original annotation files are unchanged; the swap belongs in project mapping code. Project-status line updated (dataset now stored locally). No other methodological decisions changed; D-11 remains open. |
| v0.5 | 2026-11-26 | Resolved D-11 (joint column order). The 13-column order (1 head; 2–3 shoulders; 4–5 elbows; 6–7 wrists; 8–9 hips; 10–11 knees; 12–13 ankles, 1-based) is recorded as an evidence-supported inference. It is not an explicitly documented fact: no inspected source states that list position *k* corresponds to array column *k*. New Section 6.5; Sections 6.2 and 14 updated. D-09 (left/right) is unchanged and kept separate. No other methodological decisions changed. |
| v0.6 | 2026-11-26 | D-07 split into D-07a (resolved) and D-07b (open). D-07a records the visibility semantics: 0/1 values, meaning not documented; inferred to cover occluded and out-of-frame joints; invisible coordinates unreliable. It also sets the rules for error measurement (visible ground truth only), corruption fitting (visible residuals from the fitting subset only), recogniser inputs (no visibility flags; ground-truth visibility never applied to estimated poses) and test-set use (joint eligibility for evaluation only). D-07b keeps open how invisible ground-truth coordinates enter recogniser inputs (Option A or C), to be decided with D-12 and D-20. New open item D-34: how RTMPose invalid keypoints (score ≤ 0) are handled in recogniser inputs; linked to D-07b but separate from its Option A/C choice, and undecided. D-18 remains open. Sections 5.4, 5.5, 8.2, 9.3, 12 and 14 updated. No other methodological decisions changed. |
| v0.7 | 2026-11-26 | Resolved D-12 (pose-coordinate normalisation). Primary recogniser input: per-video bounding-box normalisation, `(joint − c_v) / s_v`, where `c_v` is the median centre and `s_v` the median diagonal of that video's valid ground-truth boxes, with the same scale for x and y. The transform depends only on boxes, so it is identical for a given video in C1–C5. Within-video movement is preserved: no per-frame recentring or rescaling, rotation alignment or mirroring. Bounding boxes are recorded as supplied oracle information. Pre-registered secondary sensitivity analysis S1 (per-frame hip-midpoint centring with a per-video body scale), run only after the primary analysis. Proposed processing order recorded, dependent on open decisions. D-05, D-06, D-07b, D-19, D-20 and D-34 remain open; the proposal to express corruption residuals in per-video box-diagonal units is conditional on D-06. New Section 9.4; Sections 4.4, 5.5.3, 7.2, 8.2, 9.3, 10.6, 12 and 14 updated. No other methodological decisions changed. |
| v0.8 | 2026-11-26 | Resolved D-05 (ground-truth bounding boxes: validity, fallback and cropping). A box is valid if its row exists, its four values are finite and x2 > x1 and y2 > y1. Rows are aligned to frames from the first frame. D-12 medians use valid boxes only. Frames with an invalid box use the nearest valid box in the same video (ties: earlier frame); videos with no valid box use the full image. All frames and videos are retained. Invalid boxes are replaced before inference. Crop settings are those verified in the project's installed MMPose 1.3.2 inference pipeline (padding 1.25, then enlargement to the 3:4 input aspect ratio). Box provenance recorded as an evidence-supported inference, not a documented fact: the boxes appear to be the extent of the visible ground-truth joints, enlarged 1.2× and clipped to the image. Exact clipping boundaries uncertain. D-07a gains an explicit, documented D-05 exception to rule 4; its prohibition on direct use of ground-truth visibility flags or joint coordinates for estimated poses is unchanged. D-12 wording updated to acknowledge that its boxes carry inferred ground-truth pose information. New Sections 5.2.1–5.2.3; Sections 5.2, 5.5.2, 9.4.1, 9.4.2, 12 and 14 updated. No other methodological decisions changed. |

---

## How to read this document

Every methodological statement is tagged with one of three labels:

| Label | Meaning |
|---|---|
| **[ESTABLISHED]** | Agreed and fixed for this study. Changing it requires an explicit, recorded protocol amendment. |
| **[PROPOSED]** | The current intended approach. It may still be revised before experiments begin. |
| **[OPEN]** | Not yet decided. Each open item is listed with an identifier (e.g. `D-01`) in Section 14. |

Facts about the Penn Action dataset are taken from its official documentation and original paper. Where a detail must be verified against the downloaded files, this is stated explicitly.

The **primary experiment** is the pre-specified controlled comparison of C1–C5 described in this protocol. Any analysis outside it is labelled **secondary** or **exploratory**.

---

## 1. Project purpose and research question

### 1.1 Purpose

Skeleton-based human action recognition is commonly developed and evaluated on manually annotated (ground-truth) pose keypoints. In deployment, however, keypoints are produced by a pose estimator and therefore contain errors. This project quantifies how such errors affect downstream action recognition.

### 1.2 Research question **[ESTABLISHED]**

> How does pose-estimation error affect the performance of downstream human action recognition models?

### 1.3 Scope **[ESTABLISHED]**

This is a **controlled empirical study**. It does not propose a new pose-estimation or action-recognition architecture. Every model used is a standard, well-understood method, so that differences in performance can be attributed to the pose input rather than to architectural novelty.

---

## 2. Research objectives and hypotheses

### 2.1 Objectives **[ESTABLISHED]**

1. Measure the recognition performance obtained from ground-truth poses, as a reference.
2. Measure how performance changes when ground-truth-trained recognisers are tested on estimated poses.
3. Measure whether training on estimated poses changes this effect.
4. Measure the effect of controlled, synthetic pose corruption that is derived from realistic estimation errors.
5. Assess whether corruption-based training augmentation improves robustness to real estimation error.
6. Characterise pose-estimation error by joint, and recognition performance by action class.

### 2.2 Hypotheses **[PROPOSED]**

The following are **proposed hypotheses to be tested**. They are not established findings, and no result is assumed.

| ID | Proposed hypothesis | Primary comparison |
|---|---|---|
| H1 | Recognisers trained and tested on ground-truth poses achieve higher macro-F1 than ground-truth-trained recognisers tested on estimated poses. | C1 vs C2 |
| H2 | Training on estimated poses reduces the train–test mismatch, so that performance on estimated test poses is higher than when training on ground truth. | C3 vs C2 |
| H3 | Corrupting ground-truth test poses with training-derived error patterns degrades performance, and degradation increases with corruption severity. | C1 vs C4 (across severities) |
| H4 | Augmenting ground-truth training poses with synthetic corruption improves performance on estimated test poses relative to un-augmented ground-truth training. | C5 vs C2 |
| H5 | The magnitude of degradation differs between recogniser families (e.g. temporal CNN vs. frame-aggregated classical models). | Model × condition interaction |
| H6 *(exploratory)* | Pose-estimation error is not uniform across joints, and per-class performance changes are associated with errors in particular joints. | Per-joint error and per-class analysis |

The direction stated in each hypothesis is an expectation for testing. A null or reversed result is an equally valid outcome and will be reported as such.

**About H6.** H6 is **exploratory**. Any association found between errors in particular joints and per-class performance is correlational. It will **not** be described as evidence that errors in those joints *caused* the performance changes. Joint errors, action classes, viewpoints and occlusion are not independently manipulated in this design, so they may be confounded with one another.

---

## 3. Dataset and official split

### 3.1 Dataset **[ESTABLISHED]**

- **Name:** Penn Action dataset (Zhang, Zhu & Derpanis, 2013).
- **Official page:** https://dreamdragon.github.io/PennAction
- **Size:** 2,326 video sequences.
- **Action classes (15):** baseball pitch, baseball swing, bench press, bowling, clean and jerk, golf swing, jumping jacks, jump rope, pull-up, push-up, sit-up, squat, strumming guitar, tennis forehand, tennis serve.
- **Pose annotation:** 13 body joints per frame. The joints are the head and the left and right shoulders, elbows, wrists, hips, knees and ankles. *(The study's primary representation uses 12 of these joints; see Section 6.)*
- **Other per-video annotations (to be verified on download):** a per-joint visibility flag, a per-frame person bounding box, the action label, a camera-viewpoint label, and an official train/test flag.

### 3.2 Official split **[ESTABLISHED]**

- The dataset's official training/testing flag is preserved exactly.
- **Training partition:** 1,258 videos. **Test partition:** 1,068 videos.
- On download, the counts per partition and per class will be verified against these figures, and any discrepancy will be recorded before any further processing.

---

## 4. Training, validation and testing protocol

### 4.1 Validation split **[ESTABLISHED]**

- A **fixed, class-stratified 15%** of the official training partition is reserved for validation. That is approximately 189 of 1,258 videos; the exact number depends on per-class rounding (`D-02`).
- The remaining ~85% forms the **fitting subset**, used to train recognisers during model selection.
- After all selection decisions are frozen, the fitting and validation subsets are **recombined** into the full official training partition for final model training (Section 4.4, Stage B).
- The validation subset is generated once, with a recorded random seed, and saved as an explicit list of video identifiers. The **same** list is used for every condition, model and seed.

### 4.2 Unit of splitting **[ESTABLISHED]**

All splits are made at the **video level**, never at the frame level. All frames of a video belong to exactly one of the fitting, validation or test subsets. This prevents near-identical neighbouring frames from appearing on both sides of a split.

### 4.3 Isolation of the official test partition **[ESTABLISHED]**

The official test partition is used **only for final evaluation**. It is not used for:

- model fitting;
- hyperparameter or architecture selection;
- early stopping, epoch selection or checkpoint selection;
- choosing preprocessing, normalisation or feature design;
- designing or calibrating synthetic corruption (Section 8);
- any exploratory analysis that informs experimental design.

**[PROPOSED]** Exploratory analysis of pose error during design will be restricted to the fitting subset. Test-set pose-error statistics will be computed only after all design decisions are frozen, and only as reported outcomes (Section 10.5).

### 4.4 Primary model-selection strategy **[ESTABLISHED]**

For the **primary** controlled comparison, model development proceeds in two stages.

**Stage A: model selection (fitting and validation subsets)**

1. **Selection on C1 only.** For each model family, the hyperparameters are selected using **C1's ground-truth validation data**: models are fitted on ground-truth poses from the fitting subset and evaluated on ground-truth poses from the validation subset. Where applicable (the temporal CNN), the **training epoch count** is also selected on C1 validation.
2. **Frozen settings.** All selected hyperparameters and training settings, including the epoch count, are then **frozen** and applied unchanged in all five conditions. Conditions are **not** tuned independently in the primary experiment.
3. **Descriptive monitoring only.** During Stage A, C3 and C5 models may be trained on the fitting subset with the frozen settings, and their validation metrics (with the pose source for that condition) logged for descriptive monitoring. These metrics **must not** change hyperparameters, training duration, early stopping or checkpoint selection in the primary experiment.

**Stage B: final refitting (full official training partition)** **[ESTABLISHED]** (`D-33`, resolved)

4. **Reinitialise and refit.** Once Stage A is complete and every selection decision is frozen, the final models are **reinitialised and trained from scratch** on the **entire official training partition**: fitting plus validation subsets, all 1,258 videos.
5. **Fixed training duration.** For the temporal CNN, the **frozen epoch count** from Stage A is reused. **No validation-based early stopping or checkpoint selection** is used during final refitting, because no held-out validation data remains.
6. **Shared final model for C1, C2 and C4.** For each seed, a single final ground-truth-trained model is fitted on ground-truth poses from the full training partition. That **same final model** is evaluated on ground-truth test poses (C1), estimated test poses (C2) and corrupted ground-truth test poses (C4). Only the test-time pose input changes.
7. **Final model for C3.** C3 trains the same architecture with the frozen settings on **estimated** poses from the full official training partition.
8. **Final model for C5.** C5 trains the same architecture with the frozen settings on **ground-truth** poses from the full official training partition, with **synthetic corruption augmentation**.
9. **Data-fitted preprocessing.** Any preprocessing step whose parameters are estimated from data (for example, feature standardisation statistics) is **refitted on the full official training partition** (all 1,258 videos) for the final models. It is **never** fitted using the test partition. Pose normalisation (`D-12`, resolved, Section 9.4) estimates no parameters across videos, so it needs no refitting. Whether any data-fitted preprocessing is used, such as feature standardisation, is still open (for example under `D-19`); if it is used, this rule applies.
10. **Corruption model unchanged.** The corruption model used in C4 and C5 remains derived **exclusively from the original fitting subset** (Section 8.1). Validation residuals are not used to fit or calibrate it, even though validation videos join the final training data.
11. **Shared seeds.** The same model-initialisation and training seeds are used across comparable conditions wherever technically possible. Seed *k* for C3 and C5 uses the same seed value as seed *k* for the shared C1/C2/C4 model.
12. **Test partition untouched.** The official test partition is used in neither Stage A nor Stage B. It is used only for final evaluation of the Stage B models (Section 4.3).

**Role of the validation data.** Validation data is used **first for model selection** (Stage A). Only **after** all selection decisions are frozen does it become part of the final training data (Stage B). It cannot influence a selection decision after being used for training, because no decisions remain to be made at that point. Refitting on all 1,258 videos lets the final models use the full training partition rather than ~85% of it.

**Rationale for the selection strategy.** This strategy puts the controlled comparison first. All conditions share one set of model-selection decisions, made once on clean data. Differences between conditions can therefore be attributed to the pose input used in training or testing, rather than to different tuning outcomes. Tuning each condition independently would add a second source of variation: one condition might perform better simply because its search found more favourable hyperparameters. The trade-off is that conditions C3 and C5 run with settings chosen for ground-truth data, which may understate their best achievable performance (see Section 12).

**Secondary analysis (not part of the primary protocol) [PROPOSED].** Tuning each condition separately on its own validation data may be reported as a clearly labelled secondary analysis. It would estimate how much of any difference between conditions can be recovered by condition-specific tuning. Its results will not replace the primary comparison.

### 4.5 Data flow summary

| Subset | Source | Stage A: model selection | Stage B: final refitting | Final evaluation |
|---|---|---|---|---|
| Fitting | ~85% of official training partition | Training candidate recognisers; **sole source** of residuals for the corruption model (Section 8) | Part of final training data | Not used |
| Validation | 15% of official training partition, class-stratified | Hyperparameter and epoch selection **on C1 (ground-truth) data only**; descriptive monitoring for C3 and C5 | Part of final training data (after all decisions are frozen); **not** used for early stopping or corruption fitting | Not used |
| Full training partition | Fitting + validation (1,258 videos) | Not used as a single set | Training final models for all conditions with frozen settings | Not used |
| Test | Official test partition (1,068 videos) | Not used | Not used | **Only** use: evaluation of the final models |

---

## 5. Pose-estimation methodology

### 5.1 Estimator **[PROPOSED]**

- **Planned estimator:** pretrained **RTMPose** (Jiang et al., 2023), an MMPose-based top-down 2D pose estimator that outputs 17 keypoints in the COCO format.
- **No fine-tuning [ESTABLISHED]:** the estimator is **not** fine-tuned or adapted on Penn Action. Its errors therefore reflect an off-the-shelf estimator applied to a new domain.
- **Unconfirmed [OPEN]:** the exact RTMPose variant and model size, its training data, the checkpoint file and its checksum, the input resolution, and the inference implementation (`D-03`, `D-04`).

### 5.2 Person localisation **[ESTABLISHED]**

- Penn Action's **ground-truth bounding boxes** are used to crop the person before pose estimation.
- **Rationale:** this isolates pose-estimation error from person-detection error. The study measures the effect of keypoint error given a correct person location, not the effect of detection failures.
- **Consequence:** results represent a setting that is optimistic relative to a fully automatic pipeline (see Section 12).
- **Box validity, fallback and cropping** are resolved under `D-05` (v0.8; Sections 5.2.2 and 5.2.3).

#### 5.2.1 Box provenance **[ESTABLISHED, inferred]** (`D-05`)

The dataset documentation does not state how the bounding boxes were produced. Analysis of the **training partition only** supports the following inference. It is **not** a documented dataset fact, and it is not proven.

- **Inferred rule:** each box is the rectangle spanning the **visible** ground-truth joints of that frame (all 13 joints, head included), **enlarged 1.2× about its centre**, then **clipped to the image**.
- **Evidence:** across 87,444 training frames with at least two visible joints, this rule reproduces all four box sides **within 1 pixel in 99.86%** of frames. Agreement is at least 98.4% in every action, viewpoint and image-size group. Without clipping, agreement is 88.4%. Excluding the head gives 21.7%, and other enlargement factors (1.1, 1.15, 1.25, 1.3) give at most 0.01%. The point-shaped and inverted boxes found in the data are consistent with the rule when only one or no joint is visible.
- **Exceptions:** two training videos, **2263** and **2276** (both tennis serve), do not follow the rule (47 frames differ by more than 5 pixels). These exceptions are unexplained.
- **Uncertain:** the exact image-clipping boundaries. Clipped box sides typically lie between about 0.5 and 1 pixel inside the image edges, but the precise clipping rule has not been established.
- **Not examined:** whether test-partition boxes follow the same rule. The policy in Section 5.2.2 does not depend on it.

**What the boxes encode.** If the inference is correct, each box encodes the **overall spatial extent, centre and scale of the visible ground-truth joints** in that frame. It does **not** reveal the position of each individual joint. Because invisible joints are excluded from the extent, the box also changes when joints change visibility. The supplied boxes are therefore **privileged, ground-truth-derived localisation information** (Sections 5.5.2, 9.4.2 and 12).

#### 5.2.2 Box validity and fallback (`D-05`, resolved in v0.8) **[ESTABLISHED]**

1. **Valid box.** A box is valid if its row exists, all four values are finite (values are cast to float), and **x2 > x1 and y2 > y1**. The check uses only the box values.
2. **Row alignment.** Box rows are aligned to frames starting from the first frame. Frames without a box row are treated as having an invalid box, and surplus rows are ignored.
3. **D-12 medians** (Section 9.4.1) are computed from **valid boxes only**.
4. **Frames with an invalid box** are cropped for pose estimation using the box of the **nearest frame with a valid box in the same video**. If two such frames are equally near, the **earlier** frame is used.
5. **Videos with no valid box** use the **full image** `[0, 0, W, H]` for cropping, and the image centre and image diagonal for D-12 normalisation. Such videos are recorded and their number reported.
6. **Retention.** All frames and all videos are retained in every condition. No frame or video is dropped because of its box.
7. **Replacement before inference.** Invalid boxes are replaced (rules 4–5) **before** they are passed to the estimator. The installed MMPose inference function does not reject zero-size boxes.

All rules use only each video's own boxes, frame count and image size. They use no action labels, results or information from other videos, and they were not chosen using the test partition.

**Training-partition evidence:**
- 12 invalid boxes occur in 3 videos: point-shaped boxes in videos 0038 (7 consecutive frames) and 0600 (4 frames, at the start), and one inverted box in video 2173 (last frame).
- One video (1865) has one box row fewer than its frames. Which frame lacks the box cannot be determined, and both possible alignments fit equally well.
- Every training video has at least 18 valid boxes and at least 90% valid boxes. In the three affected videos, excluding the invalid boxes moves the D-12 centre by at most 1.2% of the median diagonal and changes the median diagonal by at most 1.6%.

#### 5.2.3 Crop settings (`D-05`, resolved in v0.8) **[ESTABLISHED]**

The crop settings are those **verified in this project's installed MMPose 1.3.2 inference pipeline** for the RTMPose-M configuration (`docs/POSE_ESTIMATION_SETUP.md`). They are not claimed to be universal across RTMPose implementations.

- The supplied box is first **padded by 1.25** about its centre.
- The padded region is then **enlarged to the 3:4 model input aspect ratio** (192 × 256). Only the shorter side is lengthened; nothing is shrunk.
- A single affine transform then crops and resizes the image to 192 × 256. Regions outside the image are filled with black.
- These settings are **not tuned** on Penn Action.

As a consequence, crops can include substantial background, especially for wide or thin boxes. Visible joints are almost always inside the crop. Invisible body parts can fall outside it, because they are excluded from the box extent.

### 5.3 Outputs to record **[PROPOSED]**

For every frame of every video:

- 17 COCO keypoint coordinates in original image pixel space;
- the per-keypoint confidence score, where the implementation provides one **[ESTABLISHED requirement]**;
- the bounding box used as input;
- the identifiers of the estimator variant, checkpoint and configuration.

Raw estimator outputs are stored before any joint mapping, so that the mapping can be revised without re-running inference.

**Confidence scores [ESTABLISHED].** In the primary experiment, RTMPose confidence scores are **not** used as inputs to action-recognition models. Ground-truth poses have no equivalent confidence values, so supplying confidence only to estimated-pose conditions would make the inputs unequal between conditions. Confidence scores are still **retained** as estimator outputs and may be used in secondary, descriptive error analysis, for example relating confidence to per-joint error.

### 5.4 Pose-estimation error measurement **[PROPOSED]**

After mapping to the **12-joint primary representation** (Section 6), per-joint error is computed against ground truth. Candidate measures include:

- the Euclidean error in pixels;
- error normalised by a defined body or box scale, such as bounding-box diagonal or torso length (`D-06`);
- the Percentage of Correct Keypoints (PCK).

**PCK definition.** PCK counts a predicted joint as correct when its distance from ground truth is below a threshold *α × s*, where *s* is a stated **reference scale** and *α* is a stated **threshold fraction**. A PCK value has no meaning without both. No result will be labelled in a form such as "PCK@0.2" unless the reference scale is stated with it, for example "PCK@0.2 of the bounding-box diagonal". The reference scale and threshold(s) remain open under `D-06`.

Only **visible** ground-truth joints are valid for error measurement. (0,0) placeholders are excluded, and frames with no valid visible ground-truth joint are unscorable (Section 5.5, `D-07a`).

Head-joint error is outside the primary analysis and may be reported only in the exploratory analysis described in Section 6.3.

### 5.5 Visibility and missing joints

Penn Action provides a per-joint visibility flag. Its handling was investigated on the downloaded annotations and is split into two decisions:
- `D-07a` covers the semantics, error measurement, corruption fitting, the use of visibility as an input, and test-set use. **[ESTABLISHED, v0.6]**
- `D-07b` covers how invisible ground-truth coordinates enter recogniser inputs. **[OPEN]**

**[ESTABLISHED]** All rules are applied **consistently across all conditions**, and are recorded before any test-set evaluation.

#### 5.5.1 What is known about visibility

**Documented facts.**
- The visibility array holds **0/1 values**, one per joint per frame (`[frames × 13]`, logical).
- The official website and README describe the annotations only as "2D locations and visibility". **No source defines what 0 and 1 mean.**
- The supplied MATLAB tool (`CreatePointLightDisplay.m`) simply omits joints with visibility 0 from its display.
- The annotation tool used to build the dataset (VATIC-ACTION) documents separate *lost* (outside the view), *occluded* and *generated* (automatically interpolated) flags in its own export format. How these became Penn Action's single flag is not documented.
- The original ICCV 2013 paper could not be accessed during this investigation.

**Empirical findings** (training partition only, `train = 1`: 1,258 videos, 87,543 frames; 78.1% of joint observations visible):
- *Self-occlusion pattern:* in side views, the far side of the body is much less often visible. Visibility of the even versus odd 1-based columns (see `D-09`) is:
  - `pose = left`: 0.604 versus 0.867;
  - `pose = right`: 0.848 versus 0.664;
  - front views: a difference of only 0.04.
- *Out-of-frame pattern:*
  - 36.4% of invisible joints lie within 5 px of the frame edge, against 0.07% of visible joints. All invisible coordinates lie inside the frame.
  - Joints typically out of shot are mostly invisible (for example, strum_guitar ankles are 3% visible).
  - Separately, 539 of 889,216 visible joints lie outside the frame, so the flag is not a clean in-frame indicator.
- *Frozen positions:* 65.6% of consecutive invisible coordinates are exactly repeated from one frame to the next (median movement 0.00 px). The figure for visible coordinates is 25.7% (median 1.33 px).
- *Distorted limb lengths* (median, distal joint invisible versus visible):
  - elbow→wrist 34.8–41.0 px versus 27.7–28.9 px;
  - hip→knee 53.1–55.2 px versus 46.0–46.8 px;
  - knee→ankle 232.7–288.9 px versus 43.0 px.
- *Placeholders:* 13 invisible joints at exactly (0,0), all in one frame of training video 2173. 87 training frames have all 13 joints invisible.
- *Stability:* 61.3% of joint tracks never change visibility; invisible stretches have a median length of 13 frames.
- *Consistency:* visibility rates vary strongly by action, but the training and test partitions match overall (0.781 versus 0.778). The test partition was used only for this descriptive comparison.

**Evidence-supported inferences** (not documented facts):
- Visibility 0 **includes both occluded and out-of-frame joints**.
- **Coordinates stored for invisible joints are unreliable.** Many are edge-clamped or frozen, and they give distorted limb lengths, so they cannot be treated as true joint positions.
- No evidence was found either way on whether visibility encodes annotator confidence.

#### 5.5.2 D-07a: rules **[ESTABLISHED, v0.6]**

1. **Pose-estimation error** (Section 5.4) is calculated only on **visible** ground-truth joints. (0,0) placeholders are excluded. Frames with no valid visible ground-truth joint are **unscorable** and are reported as such (counted, not silently dropped).
2. **Corruption distributions for C4/C5** (Section 8) are fitted only on residuals at **visible** ground-truth joints in the **fitting subset**.
3. **Visibility flags are never recogniser inputs** in any condition. They exist only for ground-truth poses, so they would make inputs unequal between conditions. They are also strongly action-specific, and could act as a shortcut unrelated to body motion.
4. **Ground-truth visibility is never applied to estimated poses** in C2, C3 or C5, whether as a mask, an imputation rule or any other preprocessing information.

   *Documented exception (`D-05`, v0.8):* the supplied ground-truth bounding boxes. They are inferred to be derived from the visible ground-truth joints (Section 5.2.1), so they **indirectly** encode ground-truth visibility and the overall extent, centre and scale of the visible ground-truth pose. They are used to crop frames for the estimator (Section 5.2.3) and to normalise poses (`D-12`, Section 9.4), identically in every condition. This exception covers the boxes only. Rule 4 continues to **prohibit any direct application of ground-truth visibility flags or ground-truth joint coordinates to estimated poses**. The resulting limitation is recorded in Section 12.
5. **Visibility-conditioned corruption** remains open under `D-18`.
6. **Test-set visibility** may be used only to decide which ground-truth joints are **eligible for evaluation**. It must not be used for model fitting, preprocessing decisions or hyperparameter selection.

#### 5.5.3 D-07b: invisible ground-truth coordinates in recogniser inputs **[OPEN]**

How invisible ground-truth coordinates enter recogniser inputs remains undecided between:
- **Option A:** retain the coordinates as released, with (0,0) placeholders handled appropriately.
- **Option C:** reconstruct invisible coordinates using a defined within-video interpolation and fallback policy.

**Why it matters.** Invisible ground-truth coordinates contain annotation artefacts (Section 5.5.1) that a recogniser could learn in ground-truth conditions, but that do not occur in estimated poses. The choice may therefore affect comparisons such as C1 versus C2.

This decision must be made together with `D-20` (temporal preprocessing), consistently with `D-12` (normalisation, resolved in v0.7, Section 9.4), and before any test-set evaluation. Because the D-12 normalisation is computed from bounding boxes rather than joints, the D-07b choice changes only the coordinate values being normalised, not the normalisation itself.

**Related but separate: RTMPose invalid keypoints (`D-34`, open).** RTMPose sets the location of any keypoint with score ≤ 0 to −1 before converting it to image coordinates (see `docs/POSE_ESTIMATION_SETUP.md`, Section 4). How such keypoints are handled in recogniser inputs is a **separate unresolved preprocessing question**. It is linked to D-07b, because both concern unreliable coordinates in recogniser inputs, but it is **not** part of the Option A/C choice above. No decision on it has been made.

---

## 6. Mapping between Penn Action and COCO joints

### 6.1 Primary joint representation: 12 joints **[ESTABLISHED]**

RTMPose outputs 17 COCO keypoints, and Penn Action annotates 13 joints. All conditions use **one common 12-joint primary representation**, made up of the joints that correspond directly in both formats: the left and right **shoulders, elbows, wrists, hips, knees and ankles**.

- The **head is excluded** from the primary representation in **every** condition, including ground-truth conditions.
- This is a **deliberate representation choice**, not missing data. No reliable correspondence has been established between any COCO keypoint and Penn Action's head annotation, and none is assumed or invented.
- The same 12-joint representation, in the same joint order, is used for recogniser inputs, error measurement and corruption in every condition.

### 6.2 Mapping table

Penn Action joint indices below follow the official numbered joint list (1-based); the 0-based column is the corresponding column of the `x`/`y`/`visibility` arrays. **Column order is resolved (`D-11`, Section 6.5) as an evidence-supported inference.** No inspected source explicitly states that list position *k* is array column *k*.

**Left/right is swapped [ESTABLISHED, v0.4].** Penn Action columns labelled LEFT hold the subject's anatomical RIGHT, and columns labelled RIGHT hold the subject's anatomical LEFT (`D-09`, Section 6.4). Each Penn Action joint is therefore mapped to the COCO keypoint for the **opposite-named** side. COCO keypoints are taken to use the subject's anatomical left/right (see the assumption in Section 6.4).

| Penn Action label (1-based index) | Penn column (0-based) | Subject's anatomical side (D-09) | COCO keypoint (0-based index) | Status |
|---|---|---|---|---|
| Head (1) | 0 | — | None assumed | **Excluded from primary representation [ESTABLISHED].** Exploratory only (`D-08`) |
| Left shoulder (2) | 1 | Right | Right shoulder (6) | Primary; swap **[ESTABLISHED]** (`D-09`); column order **[ESTABLISHED, inferred]** (`D-11`) |
| Right shoulder (3) | 2 | Left | Left shoulder (5) | Primary; swap **[ESTABLISHED]** (`D-09`); column order **[ESTABLISHED, inferred]** (`D-11`) |
| Left elbow (4) | 3 | Right | Right elbow (8) | Primary; swap **[ESTABLISHED]** (`D-09`); column order **[ESTABLISHED, inferred]** (`D-11`) |
| Right elbow (5) | 4 | Left | Left elbow (7) | Primary; swap **[ESTABLISHED]** (`D-09`); column order **[ESTABLISHED, inferred]** (`D-11`) |
| Left wrist (6) | 5 | Right | Right wrist (10) | Primary; swap **[ESTABLISHED]** (`D-09`); column order **[ESTABLISHED, inferred]** (`D-11`) |
| Right wrist (7) | 6 | Left | Left wrist (9) | Primary; swap **[ESTABLISHED]** (`D-09`); column order **[ESTABLISHED, inferred]** (`D-11`) |
| Left hip (8) | 7 | Right | Right hip (12) | Primary; swap **[ESTABLISHED]** (`D-09`); column order **[ESTABLISHED, inferred]** (`D-11`) |
| Right hip (9) | 8 | Left | Left hip (11) | Primary; swap **[ESTABLISHED]** (`D-09`); column order **[ESTABLISHED, inferred]** (`D-11`) |
| Left knee (10) | 9 | Right | Right knee (14) | Primary; swap **[ESTABLISHED]** (`D-09`); column order **[ESTABLISHED, inferred]** (`D-11`) |
| Right knee (11) | 10 | Left | Left knee (13) | Primary; swap **[ESTABLISHED]** (`D-09`); column order **[ESTABLISHED, inferred]** (`D-11`) |
| Left ankle (12) | 11 | Right | Right ankle (16) | Primary; swap **[ESTABLISHED]** (`D-09`); column order **[ESTABLISHED, inferred]** (`D-11`) |
| Right ankle (13) | 12 | Left | Left ankle (15) | Primary; swap **[ESTABLISHED]** (`D-09`); column order **[ESTABLISHED, inferred]** (`D-11`) |

Throughout the project, the 12 primary joints are named by the subject's **anatomical** side, not by Penn Action's column labels. Code should define the mapping by explicit names and check those names against the estimator's keypoint metadata. The swap must be applied in project code. It must never be applied by editing the original annotation files.

COCO keypoints not used in the primary representation: nose (0), left eye (1), right eye (2), left ear (3), right ear (4).

### 6.3 The head joint (exploratory only) **[OPEN]**

The Penn Action head annotation is **retained** in the stored data. The RTMPose/COCO output has **no keypoint labelled "head"**, and its facial keypoints (nose, eyes, ears) are **not assumed** to be equivalent to Penn Action's head annotation. That annotation's precise anatomical definition, for example head centre or top of head, must first be established by inspecting the annotations.

A separate **exploratory** head analysis will be carried out only if a defensible correspondence can be established using the **fitting subset only**. Candidate approaches to examine include a single facial keypoint, the mean of available facial keypoints, or a fitted offset from facial keypoints (`D-08`). If no defensible correspondence is found, the head analysis will not be performed, and the reason will be reported. No exploratory head result will alter the primary 12-joint experiment.

### 6.4 Left/right convention **[ESTABLISHED, v0.4]** (`D-09`, resolved)

**Adopted convention.** Penn Action columns labelled LEFT correspond to the subject's anatomical RIGHT, and columns labelled RIGHT correspond to the subject's anatomical LEFT. When Penn Action joints are mapped to COCO/RTMPose keypoints, every left/right pair is swapped: shoulders, elbows, wrists, hips, knees and ankles (Section 6.2).

**Rationale.** RTMPose outputs COCO keypoints, which follow the subject's anatomical left/right. Without the swap, every ground-truth joint would be paired with the estimator's prediction for the opposite side of the body. That would add a large, systematic, artificial error to every estimated-pose condition and to the corruption model fitted from those residuals. The swap makes ground truth and estimates refer to the same anatomical joint.

**Evidence** (training partition only, `train = 1`; no test-partition videos were used). The dataset archive was obtained from the official URL and has SHA-256 `e3e41bd99deb7b3beb9785f78b209de272bb0fa60ff4432ffa88118907a797de`.

1. *Annotation statistics.* Across training videos, the labels flip with the viewpoint label. Consider frames where both shoulders are visible, and ask whether the LEFT-labelled shoulder lies to the image right of the RIGHT-labelled shoulder:
   - `pose = front`: 3.9% of 16,604 frames (5.4% of 295 videos by majority);
   - `pose = back`: 82.1% of 11,930 frames (77.4% of 345 videos);
   - actions with a stable viewpoint: bowl, pullup and squat give 0% for front and 100% for back.

   This shows the labels follow the body, not the image side. On its own, it cannot show which anatomical side is meant.
2. *Visual inspection.* Ground-truth overlays were made for 44 frames from 15 training videos (bowl, squat, pullup and jumping_jacks; front and back views). The videos were selected by joint visibility only. Facing direction was judgeable in 7 videos:
   - front: 1037, 1739, 0543, 1157;
   - back: 0536, 0496, 1192.

   In **all 7**, the LEFT-labelled joints lie on the subject's anatomical **right**. Facing direction was judged inconclusive in 3 videos (1177, 1146, 1702), and those videos did not contradict the finding. The remaining 5 overlays (0542, 1741, 1167, 1038, 1735) were not individually assessed for facing direction.
3. *Independent review.* The researcher independently reviewed the overlays for videos 1037 (front), 1739 (front) and 1192 (back) and reached the same conclusion.

RTMPose predictions were **not** used to decide this convention.

**Assumption.** COCO/RTMPose keypoints are taken to use the subject's anatomical left/right. The MMPose COCO metadata names the keypoints by side, with matching flip pairs. One RTMPose test image of a subject facing the camera placed `left_wrist` on the image's right, which is consistent with this assumption. COCO's annotation convention was not independently re-verified in this project.

**Limitations.**
- The convention was verified visually on a **sample** of training videos (7 judgeable videos from 4 actions), not on every video.
- It assumes the convention is applied consistently across the whole dataset.
- Actions where the body rotates during the movement (for example baseball pitch, golf swing, tennis serve) were not visually inspected.
- Facing direction was judged by eye, at low resolution in some videos.
- Individual annotation errors, where one video or frame has its sides swapped, would not be corrected by this global rule. Such errors would appear as ground-truth noise.

**Data handling.** The original Penn Action annotation files are **unchanged**: all 2,326 `.mat` files were verified against SHA-256 checksums recorded before inspection. The dataset, frames and overlays are stored locally, outside the repository, and are not redistributed. The swap will be applied only by the project's mapping code, which is not yet implemented.

### 6.5 Joint column order **[ESTABLISHED, v0.5, inferred]** (`D-11`, resolved)

**Adopted column order** (1-based column of the `x`, `y` and `visibility` arrays; the 0-based index is one less):

| Columns (1-based) | Joint type |
|---|---|
| 1 | Head |
| 2–3 | Shoulders |
| 4–5 | Elbows |
| 6–7 | Wrists |
| 8–9 | Hips |
| 10–11 | Knees |
| 12–13 | Ankles |

**Status of this mapping.** It is an **evidence-supported inference, not an explicitly documented fact**. The official dataset website gives a numbered list of 13 annotated joints in this order. The annotation arrays have 13 columns. But **no source inspected in this project explicitly states that list position *k* corresponds to array column *k***. The sources inspected were:
- the official website;
- the archive's README;
- the supplied MATLAB tools (`visualize.m`, `CreatePointLightDisplay.m`, which treat all 13 columns identically);
- the website's `Keypoints.png` figure (point-light displays, not a numbered joint diagram).

**Supporting evidence** (training partition only, `train = 1`, 1,258 videos; no test-partition videos were used):
1. *Agreement with the official numbered joint list* for every column.
2. *Training-set annotation geometry*, testing body positions without using the LEFT/RIGHT names. "Upright actions" excludes bench_press, pushup and situp.
   - *Vertical order in upright actions:* column 1 above shoulder columns 2/3 in 99.0%/98.7% of frames; shoulders (2, 3) above hips (8, 9) in 99.8%/99.9%; hips above knees (10, 11) in 99.1%/98.0%; knees above ankles (12, 13) in 97.2%/98.9%.
   - *Proximity along the limb:* knee columns (10/11) lie nearer the hips than ankle columns (12/13) in 98.3%/98.7% of frames. This makes the **leg-chain order strongly supported**. Elbow columns (4/5) lie nearer the shoulders than wrist columns (6/7) in only 68.8%/69.1%, because arms frequently bend (for example bench press, pull-ups, clean and jerk). So the **elbow-versus-wrist distinction is only weakly supported by geometry**, and rests mainly on evidence items 1 and 3.
   - *Consistent pairing:* pairing even columns with even (2-4-6-8-10-12) and odd with odd (3-5-7-9-11-13) gives the shorter limb connections in 97.1–99.5% of frames across all five limb segments. Each pair is therefore ordered consistently, with the even columns forming one body side and the odd columns the other.
3. *Visual inspection:* ground-truth overlays were made for 44 frames from 15 training videos (bowl, squat, pullup, jumping_jacks). Ten of these overlays (29 frames) were individually viewed. In all 10, each labelled joint lay on the corresponding body part and the limb chains were anatomically coherent. The researcher independently reviewed three of these overlays (videos 1037, 1739, 1192). The figure of seven judgeable videos reported in Section 6.4 refers to facing direction for D-09, not to joint placement.

No inspected evidence contradicts this column order.

**Relationship to D-09.** D-11 establishes **which type of joint** each column holds, and that each left/right pair is ordered consistently. D-09 (Section 6.4) separately establishes **which anatomical side** each group of paired columns represents: columns labelled LEFT, which are the even 1-based columns, hold the subject's anatomical RIGHT. The two decisions rest on different evidence and together give the complete mapping in Section 6.2. Neither decision's evidence is used to prove the other.

**Limitations.**
- The original ICCV 2013 paper could not be accessed during this investigation; the publisher's site refused automated access. Whether it contains a numbered joint diagram is unknown.
- The column-to-joint correspondence is inferred, not stated by the dataset authors.
- The elbow-versus-wrist distinction has weaker geometric support than the other joints.
- Individual annotation errors, such as a mislabelled joint in a particular video or frame, are possible. They would act as ground-truth noise and are not ruled out by this dataset-level decision.
- The precise anatomical meaning of the head point (for example head centre or top of head) is a separate question under `D-08` and is not resolved here.

**Data handling.** As for D-09, the original annotation files are unchanged, and the analyses and overlays are stored locally, outside the repository.

---

## 7. Experimental conditions C1–C5 **[ESTABLISHED]**

The five conditions are fixed and are reproduced exactly:

- **C1:** Ground-truth poses for training and testing.
- **C2:** Ground-truth poses for training and estimated poses for testing.
- **C3:** Estimated poses for training and testing.
- **C4:** Ground-truth poses for training and artificially corrupted testing poses.
- **C5:** Ground-truth poses with corruption augmentation for training, and estimated poses for testing.

### 7.1 Summary table

| Condition | Training poses | Test poses | Final model (per seed) | Role in the study |
|---|---|---|---|---|
| C1 | Ground truth | Ground truth | Shared final ground-truth-trained model | Reference performance |
| C2 | Ground truth | Estimated (RTMPose) | Same shared final model as C1 | Effect of real estimation error at test time |
| C3 | Estimated (RTMPose) | Estimated (RTMPose) | Separately trained; frozen settings | Effect of matching training data to test-time error |
| C4 | Ground truth | Ground truth + synthetic corruption | Same shared final model as C1 | Controlled, severity-graded effect of error |
| C5 | Ground truth + synthetic corruption augmentation | Estimated (RTMPose) | Separately trained; frozen settings | Whether synthetic augmentation improves robustness to real error |

All poses use the 12-joint primary representation (Section 6.1). All final models are trained on the full official training partition (1,258 videos) with settings frozen in Stage A (Section 4.4).

### 7.2 Controls common to all conditions **[ESTABLISHED]**

- Identical fitting, validation and test video lists.
- The same final training data: the full official training partition, with the pose source for that condition.
- The same recogniser families and architectures.
- **Hyperparameters and training duration selected once on C1 validation, then frozen** for all conditions (Section 4.4).
- Final models trained from scratch with the frozen epoch count and no validation-based early stopping.
- The same model-initialisation seeds across comparable conditions, wherever technically possible.
- The same 12-joint primary representation and joint order.
- No confidence scores as recogniser inputs (Section 5.3).
- Consistent treatment of visibility and missing joints (Section 5.5).

**[ESTABLISHED]** The same per-video pose normalisation is applied in every condition, and a given video receives the identical transform in C1–C5 (`D-12`, Section 9.4).

**[PROPOSED]** The same temporal preprocessing (`D-20`) is applied in every condition, and all preprocessing steps, including corruption and normalisation, follow the fixed, documented order in Section 9.4.3.

### 7.3 Validation data in C3 and C5 (resolved) **[ESTABLISHED]**

Validation data with the condition's own pose source (estimated poses for C3; ground-truth poses with or without corruption for C5) does **not** determine model selection in the primary experiment. It may be logged for descriptive monitoring only, during Stage A (Section 4.4). In Stage B the validation videos are part of the training data, so no validation monitoring takes place. Which descriptive validation metrics are logged will be recorded in the run configuration (`D-13`, resolved).

### 7.4 Test-time corruption sampling in C4 **[PROPOSED]**

For each severity level, C4 uses **multiple fixed corruption draws**, each generated from its own recorded seed, rather than a single random draw. Each final ground-truth-trained model (shared with C1 and C2) is evaluated on every draw. Repeated draws make it possible to separate the effect of **corruption severity** from the **random sampling variation** of any one particular corrupted test set. The number of draws per severity remains open (`D-14`).

---

## 8. Artificial corruption methodology

### 8.1 Principles **[ESTABLISHED]**

- C4 and C5 use synthetic pose corruption.
- Corruption is applied to **ground-truth** poses in the 12-joint primary representation.
- Corruption is derived from realistic error patterns, measured as RTMPose residuals against ground truth.
- **Residuals come from the fitting subset only.** Residuals from the validation subset and the official test partition are **never** used to estimate, calibrate or select the corruption distribution or its severity levels (`D-10`, resolved). This remains true in the final refitting stage. Although C5's final model is trained on all 1,258 training videos, the corruption model applied to them is the one derived from the fitting subset (Section 4.4, Stage B).
- C4 evaluates a ground-truth-trained recogniser on corrupted ground-truth test poses.
- C5 uses corrupted ground-truth pose augmentation during training, and estimated poses during testing.

**Rationale for using the fitting subset only.** This is a conservative choice. The validation subset is used for model selection, and the test partition is used for final evaluation. Keeping both entirely separate from corruption design ensures that neither has influenced the error model that C4 and C5 depend on, so neither can contribute to an optimistic bias in their results.

### 8.2 Proposed procedure **[PROPOSED]**

1. Run RTMPose on the **fitting subset** and map the output to the 12-joint primary representation (Sections 5–6).
2. Compute residuals (estimated minus ground truth) per joint and per frame, in a normalised coordinate frame (`D-06`). *Proposal, conditional on D-06's decision:* express residuals in units of the per-video box diagonal `s_v` used for normalisation (Section 9.4.3). Only residuals at **visible** ground-truth joints in the fitting subset are eligible. (0,0) placeholders are excluded (`D-07a`, Section 5.5).
3. Fit or summarise an error model from these residuals. Candidates are listed in `D-15`:
   - independent per-joint Gaussian or empirical distributions;
   - per-joint models that account for direction and correlation between the x and y coordinates;
   - models that also capture temporal correlation of errors within a video;
   - models that capture occasional large failures, such as outliers or left/right swaps.
4. Define a set of severity levels, for example by scaling the fitted error model (`D-16`).
5. **C4:** apply corruption at each severity level to ground-truth test poses, using multiple fixed, recorded seeds per severity (Section 7.4, `D-14`).
6. **C5:** apply corruption stochastically to ground-truth training poses during training, with an augmentation probability and severity range to be decided (`D-17`). The augmentation settings are chosen without reference to C5 validation or test performance.

### 8.3 Realism check **[PROPOSED]**

As a descriptive diagnostic, which does not change the conditions, the distribution of synthetic residuals can be compared with that of real RTMPose residuals **within the fitting subset**. Comparing C4 and C2 performance at the end of the study is a reported outcome. It may **not** be used to retune the corruption model after test results are seen.

### 8.4 Unresolved questions **[OPEN]**

- The family of corruption distributions (`D-15`).
- The number and definition of severity levels (`D-16`).
- The augmentation probability and severity schedule for C5 (`D-17`).
- The number of C4 corruption draws per severity (`D-14`).
- Whether corruption accounts for joint visibility (`D-18`).

---

## 9. Downstream action-recognition models

### 9.1 Planned model families **[ESTABLISHED]**

| Model | Role | Input form **[OPEN]** |
|---|---|---|
| Majority-class baseline | Lower bound; chance-level reference given class imbalance | None (uses training label distribution only) |
| Logistic regression | Linear classical model | Fixed-length per-video feature vector built from the 12-joint representation (`D-19`) |
| Random forest | Non-linear classical model | Fixed-length per-video feature vector built from the 12-joint representation (`D-19`) |
| Small temporal CNN | Model that operates over time | Fixed- or variable-length sequence of 12-joint coordinates (`D-20`) |

### 9.2 Model selection and consistency across conditions **[ESTABLISHED]**

- The same model families and architectures are used in every condition.
- **Hyperparameters, and the training duration where applicable, are selected on C1 ground-truth validation data only, then frozen across all five conditions** (Section 4.4).
- No condition is tuned independently in the primary experiment.
- Condition-specific tuning may appear only as a labelled secondary analysis.
- **Final models are reinitialised and trained from scratch on the full official training partition** (1,258 videos) with the frozen settings. The temporal CNN uses the frozen epoch count, without validation-based early stopping (Section 4.4, Stage B; `D-33`, resolved).
- Recogniser inputs contain **joint coordinates only**. RTMPose confidence scores are not used as inputs (Section 5.3; `D-23`, resolved).

### 9.3 Unspecified details **[OPEN]**

- Exact temporal CNN architecture: depth, kernel sizes, channels, pooling and regularisation (`D-21`).
- Hyperparameter search spaces, search method (such as grid or random search) and budget, all applied on C1 validation (`D-22`).
- Temporal preprocessing for sequences of different lengths: resampling to a fixed length, padding or cropping (`D-20`).
- *(Pose normalisation is now resolved: `D-12`, Section 9.4.)*
- Representation of invisible ground-truth joints in model input (`D-07b`, Section 5.5.3). Visibility flags are never inputs, and ground-truth visibility is never applied to estimated poses (`D-07a`).
- Representation of RTMPose keypoints with score ≤ 0 in model input (`D-34`, Section 5.5.3).
- Feature engineering for classical models, such as summary statistics over time, velocities or joint angles (`D-19`).
- Class-imbalance handling, such as class weighting (`D-24`).

### 9.4 Pose-coordinate normalisation (`D-12`, resolved in v0.7) **[ESTABLISHED]**

#### 9.4.1 Primary representation

Recogniser inputs use **per-video bounding-box normalisation** of the 12-joint primary representation (Section 6; mapping per `D-09` and `D-11`). For each video *v*, using that video's **valid** ground-truth bounding boxes:

- **Centre** `c_v`: the median, over valid boxes, of the box centres `((x1 + x2) / 2, (y1 + y2) / 2)`.
- **Scale** `s_v`: the median, over valid boxes, of the box diagonals `sqrt((x2 − x1)² + (y2 − y1)²)`. The **same scale is used for x and y**, so body proportions and angles are preserved.
- **Transform:** `normalised_joint = (joint_coordinate − c_v) / s_v`.

What counts as a valid box, and the fallback for a video with no valid box (image centre and image diagonal), are defined by `D-05` (resolved in v0.8, Section 5.2.2).

Rules:

1. **Identical across conditions.** The transform depends only on the video's ground-truth boxes. It is never computed directly from joint coordinates (ground-truth or estimated). The same per-video transform is therefore used in C1–C5. The boxes themselves are inferred to be derived from the visible ground-truth joints (Section 5.2.1), so the transform indirectly reflects the extent of the visible ground-truth pose, identically in every condition.
2. **Within-video movement is preserved.** There is **no** per-frame recentring, per-frame rescaling, rotation alignment or mirroring.
3. **No parameters fitted across videos.** Each video is normalised using only its own boxes. Nothing is fitted on the training partition, so nothing is refitted in Stage B, and nothing is estimated from the test partition.

**Supporting evidence** (empirical findings, training partition only, 1,258 videos; no test data used):

- The per-video median box height ranges from 77 px to 294 px (5th to 95th percentile), a ratio of about 3.8. Raw pixel coordinates would therefore mix in large differences in subject scale.
- Within videos, the box diagonal is the most stable box measure: median coefficient of variation 0.076, against 0.117 for box height and 0.151 for box width.
- All four torso joints are visible in a median of only 36% of frames per video. A body-centred reference would often depend on invisible ground-truth coordinates, which are unreliable (Section 5.5.1).

#### 9.4.2 Bounding boxes as supplied oracle information

Ground-truth bounding boxes are **supplied oracle information** in this study's design (Section 5.2). At test time, each test video is normalised using **that test video's own ground-truth boxes**. This uses the same box information that is already given to the pose estimator in C2, C3 and C5. It does **not** fit any model parameters, and it does **not** use action labels, other test videos or test results. However, the boxes **carry inferred ground-truth pose information**: the overall extent, centre and scale of the visible ground-truth joints (Section 5.2.1). Because the normalisation uses per-video medians rather than each frame's box, this information enters only in aggregated form, and identically in every condition. This use of the boxes is covered by the documented `D-05` exception to `D-07a` rule 4 (Section 5.5.2). As with box-based cropping, this makes the setting optimistic relative to a fully automatic pipeline (Section 12).

#### 9.4.3 Processing order **[PROPOSED]**

The proposed order of preprocessing steps is listed below. Where a step depends on an open decision, its content, and if necessary its position, will be confirmed when that decision is made.

1. **Joint mapping** to the 12-joint primary representation (`D-09`, `D-11`).
2. **Unreliable coordinates:** invisible ground-truth coordinates (`D-07b`, open; ground-truth poses only) and RTMPose invalid keypoints (`D-34`, open; estimated poses only). This step is placed before normalisation so that placeholder or sentinel locations are dealt with before they are treated as image positions. How either is handled is not decided here.
3. **Corruption** (C4 and C5 only; `D-14`–`D-18`, open).
4. **Per-video normalisation** (`D-12`, this section).
5. **Temporal preprocessing** (`D-20`, open). Corruption is placed before this step because any temporally correlated error model (`D-15`) is defined at the original frame rate.
6. **Feature construction and any data-fitted standardisation** (`D-19`, open; Section 4.4, Stage B, item 9).

**Corruption and normalisation.** The normalisation is a per-video translation plus a uniform scaling. If a corruption residual `e` is expressed in units of the same per-video scale `s_v`, then adding `e · s_v` in pixel space before normalisation gives exactly the same result as adding `e` after it: `((p + e · s_v) − c_v) / s_v = (p − c_v) / s_v + e`. In that case the two steps can be applied in either order. **This equivalence holds only when corruption is expressed in compatible per-video scale units.** It does not hold in general, for example for residuals expressed in pixels or in a per-frame scale. Expressing residuals in units of `s_v` is a **proposal conditional on `D-06`**, which remains open. Until D-06 is decided, the order above applies as listed.

#### 9.4.4 Pre-registered sensitivity analysis S1 (secondary)

- **Representation:** per-frame **hip-midpoint centring** with a **per-video body scale**.
- **Computed from the pose actually supplied to the model:** ground truth in C1, estimated poses in C2, C3 and C5 testing, and corrupted ground truth in C4. A reference derived from ground truth is **never** applied to estimated poses, consistent with `D-07a` rule 4.
- **Timing:** run only **after the primary analysis is complete**, with all primary settings frozen (no retuning). It is reported as a secondary analysis and does not replace the primary result.
- **Purpose:** to test whether conclusions depend on the coordinate representation. Under S1, estimator error in the hip joints also shifts the reference point, so differences between S1 and the primary result are themselves informative.
- **Not yet specified, and not implemented:** the exact body-scale definition, and the handling of frames where the hip joints are invisible or invalid. These will be fixed before S1 is run, consistently with the eventual `D-07b` and `D-34` decisions.

---

## 10. Evaluation metrics and statistical analysis

### 10.1 Metrics **[ESTABLISHED]**

- **Primary metric:** macro-averaged F1 over the 15 classes. It is chosen because it weights each class equally regardless of class frequency.
- **Secondary metric:** accuracy.
- Both metrics are computed at the **video level**: one prediction per test video.

### 10.2 Training seeds **[ESTABLISHED / PROPOSED]**

- **[ESTABLISHED]** Training is repeated with **multiple random seeds** for each model and condition, with seeds shared across comparable conditions (Section 4.4).
- **[PROPOSED]** The proposed starting point is **five training seeds**. Final confirmation remains open (`D-25`).
- For deterministic procedures where the seed has little or no effect, such as some logistic-regression solvers, this will be noted rather than reporting identical repeats as independent evidence.

### 10.3 Paired comparisons between conditions **[ESTABLISHED]**

All conditions are evaluated on the **same test videos**, so comparisons between conditions use **paired** evaluation. The planned bootstrap procedure for comparing conditions is:

1. resample test videos **with replacement within each action class**, a stratified bootstrap;
2. use the **same resampled video identifiers** for every condition in the comparison;
3. recalculate macro-F1 for each condition on each resample;
4. compute the **paired difference** in macro-F1 for each resample;
5. report the paired difference and its confidence interval.

**[OPEN]** The number of bootstrap resamples (`D-26`) and the confidence-interval method, such as percentile or BCa (`D-27`).

### 10.4 Sources of variability **[ESTABLISHED]**

Two sources of variability are kept **distinct**:

- **training-seed variability:** variation across models trained with different seeds;
- **test-video sampling variability:** uncertainty because the test set is a finite sample, estimated by the bootstrap.

**[PROPOSED]** Both are reported separately, for example a bootstrap confidence interval per seed alongside the spread across seeds. They are combined into a single interval only if a clearly specified combined procedure is adopted and documented in advance (`D-27`).

For C4, corruption-draw variability (Section 7.4) is a third source. It is reported separately in the same way.

### 10.5 Planned comparisons and multiple testing **[PROPOSED]**

- The family of planned confirmatory comparisons is pre-specified by hypotheses H1–H5. H6 is exploratory.
- The approach to multiple comparisons, whether a correction method or explicit labelling as exploratory, is to be decided (`D-28`).
- Effect sizes (paired differences in macro-F1 with confidence intervals) are prioritised over significance tests alone.

### 10.6 Secondary and exploratory analyses

**[ESTABLISHED]**

- Per-class performance (per-class F1, confusion matrices) for each condition.
- Pose-estimation error by joint for the 12 primary joints (Section 5.4). During design this is computed on the fitting subset; the test partition is analysed only after all design decisions are frozen.
- Normalisation sensitivity analysis S1: per-frame hip-midpoint centring with a per-video body scale, run only after the primary analysis with primary settings frozen (Section 9.4.4, `D-12`).

**[PROPOSED]** Exploratory analyses that are labelled as such:

- associations between joint errors and per-class performance (H6), which are correlational and not causal;
- the relationship between per-video pose error and recognition correctness;
- relationships between confidence scores and per-joint error (descriptive);
- breakdowns by camera viewpoint label;
- breakdowns by joint visibility;
- the exploratory head analysis (Section 6.3), if performed;
- condition-specific tuning (Section 4.4), as a secondary analysis.

---

## 11. Reproducibility and experiment logging

### 11.1 Environment (recorded) **[ESTABLISHED]**

Recorded in `environment-info.txt` and `requirements.txt` at the time of writing:

- Python 3.11.9; PyTorch 2.15.0+cu134; CUDA 13.4;
- NVIDIA GeForce RTX 3060 Ti (8 GB), driver 617.42;
- Windows 11 Home (build 26300), 64-bit; pip 26.3;
- relevant libraries include NumPy 2.4.6, pandas 3.1.1, SciPy 1.17.1, scikit-learn 1.9.1, OpenCV 5.1.0.95, PyYAML 6.0.3, matplotlib 3.11.2 and seaborn 0.13.2.

**Note:** `requirements.txt` does not yet include any pose-estimation inference dependency, such as an MMPose-based stack or a standalone RTMPose runtime. This depends on `D-04` and must be added and pinned before pose estimation begins.

### 11.2 Proposed logging practice **[PROPOSED]**

For every run, record:

- the **stage** of the run: Stage A (selection: fitting + validation subsets) or Stage B (final refit, full training partition);
- the condition, model, seed, corruption severity and corruption-draw seed (if applicable);
- the full configuration file, stored under `configs/`;
- the **frozen hyperparameters and epoch count** selected on C1 validation, with a reference to the Stage A selection run that produced them;
- for Stage B runs: confirmation that the model was reinitialised and trained on all 1,258 training videos, the number of epochs actually run, and that no early stopping was applied;
- for C1, C2 and C4: the identifier of the **shared final model** evaluated, so that each evaluation can be traced to one model;
- the Git commit hash of the code used, and whether the working tree was clean;
- the identifiers and checksums of the split lists;
- the estimator variant, checkpoint checksum and joint-mapping version (for estimated-pose conditions);
- the identifier and version of the corruption model (for C4 and C5);
- Stage A validation metrics: those used for C1 selection, and descriptive monitoring for C3 and C5;
- final test metrics, and per-video predictions to allow paired bootstrap analysis;
- wall-clock time and hardware.

Proposed repository locations, consistent with the existing structure:

- configurations in `configs/`;
- run records in `experiments/`;
- logs in `logs/`;
- model checkpoints in `checkpoints/` (not committed to Git);
- aggregated tables and figures in `results/tables/` and `results/figures/`.

### 11.3 Determinism **[PROPOSED]**

- Seed all random number generators: Python, NumPy, PyTorch and scikit-learn.
- Record whether deterministic GPU algorithms were enabled.
- Acknowledge that complete bit-for-bit GPU determinism may not be achievable. Seed repetition (Section 10.2) is the primary safeguard.

### 11.4 Protocol versioning **[PROPOSED]**

This protocol is version-controlled. Any change after experiments begin is recorded as a dated amendment, with a justification, and it states whether the change was made before or after test results were seen.

---

## 12. Limitations and threats to validity

**Internal validity**

- **Ground-truth bounding-box crops** remove detection error, so the results likely underestimate the degradation that a fully automatic pipeline would show. The same oracle boxes are also used for pose normalisation (Section 9.4.2), so the recognisers never face errors in person localisation or scale.
- **Privileged, ground-truth-derived localisation:** the boxes are inferred to be derived from the visible ground-truth joints (Section 5.2.1). The estimator therefore receives the overall extent, centre and scale of the visible ground-truth pose through its crop, although not the position of each individual joint. Estimated-pose errors, and differences such as C1 versus C2, are therefore likely to be smaller than with a person detector. The findings concern **pose estimation under oracle-box conditions**, not a fully automatic person-detection pipeline. Because the same boxes are used in every condition, the comparisons between conditions remain internally consistent. This use is a documented exception to `D-07a` rule 4 (Section 5.5.2).
- **Camera motion in normalised coordinates:** per-video normalisation preserves within-video movement, but it cannot separate subject movement from camera panning. Camera motion therefore remains in the inputs. It is identical across conditions for a given video (Section 9.4).
- **Model selection on C1 only:** hyperparameters and training duration chosen on clean ground-truth data may not suit C3 and C5. Their primary-experiment performance may therefore understate what condition-specific tuning could achieve. The optional secondary analysis (Section 4.4) addresses this.
- **Ground-truth annotation noise:** Penn Action annotations are themselves imperfect, so "error" is measured relative to a noisy reference.
- **Invisible joints** have unreliable ground-truth coordinates: many are edge-clamped or frozen (Section 5.5.1). They are excluded from error measurement and corruption fitting (`D-07a`). As a result, estimator accuracy on occluded or out-of-frame joints is not measured against ground truth, and the corruption model is fitted only on visible joints. Their treatment in recogniser inputs (`D-07b`) may affect ground-truth versus estimated-pose comparisons.
- **Synthetic corruption fidelity:** the corruption model may not capture all properties of real estimator errors, such as temporal correlation, structured failures or left/right swaps. This limits how far C4 results generalise.
- **Frozen hyperparameters transferred to a larger training set:** all hyperparameters, including the epoch count for the temporal CNN, are selected while training on ~85% of the training partition, then reused when refitting on 100%. The optimal values on the larger set may differ slightly. For example, the effective strength of regularisation can depend on the number of training samples, and the best training duration may change. No validation check is possible during refitting. This applies equally to all conditions.
- **Smaller corruption-fitting sample:** using the fitting subset only (~85% of training videos) gives slightly less data for estimating the error model, which is accepted in exchange for strict data separation.
- **Multiple sources of variance:** training seeds, corruption draws and test-set sampling all affect the precision of comparisons (Section 10.4).

**External validity**

- **A single dataset:** Penn Action contains mostly single-person, sports and exercise footage, so the conclusions may not transfer to other action domains, multi-person scenes or 3D settings.
- **A single estimator:** the results characterise one pretrained RTMPose configuration. Other estimators may produce different error patterns.
- **Restricted model families:** the recognisers are deliberately simple. Larger skeleton-based models, such as graph convolutional networks, may respond to pose error differently.

**Construct validity**

- **Joint-set reduction:** the primary representation uses 12 of Penn Action's 13 joints, and 12 of RTMPose's 17 keypoints. Head and facial information is excluded, which may disadvantage actions where head position is informative. This affects all conditions equally.
- **No confidence inputs:** excluding confidence scores keeps the inputs symmetric, but it means the study does not measure how much confidence-aware recognisers could mitigate estimation error.
- **Metric choice:** macro-F1 on 15 classes with modest per-class test counts may have wide confidence intervals.
- **Correlational joint analysis:** joint–class associations (H6) cannot establish causation.

---

## 13. Data handling and licensing considerations

**[ESTABLISHED]**

- Penn Action files, and any **derived annotations or estimated poses**, are **not redistributed** through this repository (consistent with `README.md`).
- The repository's `.gitignore` excludes `data/raw/`, `data/processed/`, `data/poses/`, `checkpoints/`, logs, model-weight files and common dataset or archive formats.
- Only code, configuration, documentation and aggregate results (tables and figures) are intended for publication.

**[OPEN]**

- The Penn Action terms of use must be read from the official dataset page and recorded here, including any citation requirement and restrictions on derived data (`D-29`).
- The licence and terms of the RTMPose/MMPose code and pretrained weights, and of the datasets those weights were trained on, must be checked and recorded (`D-30`).
- A licence for this repository's own code has not yet been chosen (`D-31`).
- The study will consider whether published figures may include dataset video frames, or only skeleton plots and aggregate statistics (`D-32`).

---

## 14. Decisions still requiring confirmation

### 14.1 Resolved decisions

| ID | Topic | Resolution |
|---|---|---|
| D-05 *(v0.8)* | Ground-truth bounding boxes: validity, fallback and cropping | **Valid box:** row exists, four finite values, x2 > x1 and y2 > y1 (box values only). Rows aligned to frames from the first frame; frames without a row are invalid; surplus rows ignored. D-12 medians use valid boxes only. Invalid-box frames use the nearest valid box in the same video (ties: earlier frame); videos with no valid box use the full image `[0, 0, W, H]` (and the image centre and diagonal for D-12), and are reported. All frames and videos retained. Invalid boxes replaced before inference. Crop settings as verified in the project's installed MMPose 1.3.2 pipeline: padding 1.25, then enlargement to the 3:4 input aspect ratio; not tuned. *Evidence-supported inference, not a documented fact:* boxes are the visible-joint extent (head included) enlarged 1.2× and clipped to the image; reproduces 99.86% of training boxes within 1 px; exceptions 2263 and 2276 unexplained; clipping boundaries uncertain. Boxes therefore carry privileged ground-truth-derived localisation: a documented exception to D-07a rule 4, and a limitation (Sections 5.2, 5.5.2, 12). |
| D-12 *(v0.7)* | Pose normalisation and ordering | **Per-video bounding-box normalisation:** `(joint − c_v) / s_v`, where `c_v` is the median centre and `s_v` the median diagonal of that video's **valid** ground-truth boxes, with the same scale for x and y. The transform depends only on boxes, so it is identical for a given video in C1–C5. Within-video movement is preserved: no per-frame recentring or rescaling, rotation alignment or mirroring. Boxes are supplied oracle information; test videos are normalised with their own ground-truth boxes, with no model parameters fitted and no labels used. The boxes carry inferred ground-truth pose information (extent, centre and scale of the visible joints; Section 5.2.1), entering only through per-video medians, under the D-05 exception to D-07a rule 4. Box validity and fallback rules: D-05 (resolved, v0.8). Proposed processing order recorded (Section 9.4.3), dependent on D-07b, D-34, D-14–D-18, D-20 and D-19. Corruption and normalisation are interchangeable in order **only** if residuals use compatible per-video scale units; expressing them in units of `s_v` is a proposal conditional on D-06. Pre-registered secondary sensitivity analysis S1: per-frame hip-midpoint centring with a per-video body scale, computed from the pose supplied to the model, run only after the primary analysis (Section 9.4). |
| D-07a *(v0.6)* | Visibility semantics and use | Values are 0/1, and their meaning is not documented. *Inferred:* 0 includes occluded and out-of-frame joints, and invisible coordinates are unreliable (edge-clamped, frozen, distorted limb lengths). Rules: error measured on visible ground-truth joints only, excluding (0,0) placeholders, with frames lacking any valid visible joint reported as unscorable; corruption fitted on visible residuals from the fitting subset only; visibility flags never recogniser inputs; ground-truth visibility never applied to estimated poses (C2, C3, C5), with one documented exception (v0.8): the supplied bounding boxes, which indirectly encode visible ground-truth pose information (D-05; direct use of visibility flags or joint coordinates remains prohibited); test-set visibility used only for evaluation eligibility. Visibility-conditioned corruption remains under D-18 (Section 5.5). |
| D-11 *(v0.5)* | Joint column order | 1-based columns: 1 head; 2–3 shoulders; 4–5 elbows; 6–7 wrists; 8–9 hips; 10–11 knees; 12–13 ankles. This is an **evidence-supported inference**, not a documented fact: the official website lists the 13 joints in this order, but no inspected source states that list position *k* is array column *k*. Evidence: agreement with the official list; geometry across all 1,258 training videos (leg chain strongly supported; elbow-versus-wrist weaker because arms bend); visual inspection of 10 overlays (29 frames), 3 independently reviewed by the researcher. Kept separate from D-09 (left/right). Limitations: paper not accessible; individual annotation errors possible; head-point meaning remains under D-08 (Section 6.5). |
| D-09 *(v0.4)* | Left/right convention | Penn Action columns labelled LEFT hold the subject's anatomical RIGHT, and columns labelled RIGHT hold the anatomical LEFT. Every left/right pair (shoulders, elbows, wrists, hips, knees, ankles) is swapped when mapping to COCO/RTMPose. Evidence: training-partition annotation statistics, plus visual inspection of 15 training videos (7 judgeable, all consistent; 3 independently confirmed by the researcher). RTMPose was not used for the decision. Limitation: verified on a sample, not every video. The original annotation files are unchanged; the swap is applied in project code (Sections 6.2, 6.4). |
| D-33 *(v0.3)* | Final refit | Final models are **reinitialised and trained from scratch on the full official training partition** (fitting plus validation, 1,258 videos) after all Stage A selection decisions are frozen. The temporal CNN reuses the frozen epoch count, without validation-based early stopping. C1, C2 and C4 share one final model per seed; C3 and C5 are trained on the full partition with their own pose sources. The corruption model remains derived from the fitting subset only (Section 4.4, Stage B). |
| D-10 *(v0.2)* | Residual source | Corruption model derived from the **fitting subset only**. Validation and test residuals are never used to estimate, calibrate or select it (Section 8.1). |
| D-13 *(v0.2)* | Validation pose source in C3 and C5 | Primary model selection uses **C1 ground-truth validation only**, with settings frozen across conditions. C3 and C5 validation is descriptive monitoring only (Sections 4.4, 7.3). |
| D-23 *(v0.2)* | Confidence as recogniser input | **Not used** as recogniser inputs in the primary experiment. Retained as estimator outputs for secondary descriptive analysis (Section 5.3). |
| D-08 *(primary part, v0.2)* | Head in primary representation | **Excluded** from the 12-joint primary representation in every condition (Section 6.1). The exploratory head mapping remains open below. |

### 14.2 Open

| ID | Topic | Decision required |
|---|---|---|
| D-01 | Dataset verification | Confirm that the downloaded files match 2,326 videos, 15 classes, 13 joints and the 1,258/1,068 split; record any discrepancy. |
| D-02 | Validation split | Stratification and rounding rule; random seed; file format for storing video identifier lists. |
| D-03 | RTMPose variant | Model size, its training data, input resolution, and checkpoint name and checksum. |
| D-04 | Inference implementation | Software stack and version for running RTMPose; dependency pinning in `requirements.txt`. |
| D-06 | Error normalisation and PCK | Coordinate frame and reference scale for residuals and error metrics; PCK reference scale *s* and threshold fraction(s) *α*. Proposal under consideration (v0.7): express corruption residuals in units of the per-video box diagonal `s_v` used by D-12 (Sections 8.2, 9.4.3). Not decided. |
| D-07b | Invisible ground-truth coordinates in recogniser inputs | Choose between Option A (retain the coordinates as released, handling (0,0) placeholders appropriately) and Option C (reconstruct invisible coordinates with a defined within-video interpolation and fallback policy). Decide together with D-20, consistently with D-12 (resolved, v0.7), before any test-set evaluation. The rule must be consistent across conditions. Concerns Penn Action ground-truth coordinates only; RTMPose invalid keypoints are the separate question D-34 (Section 5.5.3). |
| D-08 *(exploratory part)* | Exploratory head mapping | Whether a defensible COCO-to-head correspondence exists (assessed on the fitting subset only), and if so which one. Otherwise the head analysis is not performed. |
| D-14 | C4 corruption draws | Number of fixed corruption draws per severity level. Multiple draws are favoured (Section 7.4). |
| D-15 | Corruption distribution | The family of error models: independent, correlated, temporal, with outliers or swaps. |
| D-16 | Severity levels | The number of severity levels and how each is defined. |
| D-17 | C5 augmentation | Augmentation probability, severity range or schedule. |
| D-18 | Visibility in corruption | Whether corruption is conditioned on joint visibility. |
| D-19 | Classical-model features | The per-video feature representation for logistic regression and random forest. |
| D-20 | Temporal preprocessing | Handling of variable sequence lengths (resampling, padding, cropping) and frame rate. |
| D-21 | Temporal CNN architecture | Exact layers, sizes and regularisation. |
| D-22 | Tuning procedure | Search spaces, search method, budget and epoch-selection rule, applied on C1 validation. |
| D-24 | Class imbalance | Class weighting or other handling. |
| D-25 | Number of seeds | Final number of training seeds; five is the proposed starting point. |
| D-26 | Bootstrap count | Number of stratified, paired bootstrap resamples. |
| D-27 | CI methodology | The interval type (e.g. percentile or BCa), and whether any combined seed-and-sample procedure is adopted. |
| D-28 | Multiple comparisons | The correction method, or explicit labelling of exploratory comparisons. |
| D-29 | Dataset terms | Record the Penn Action terms of use and citation requirements. |
| D-30 | Estimator licensing | Record the RTMPose/MMPose code and weight licences. |
| D-31 | Repository licence | Choose a licence for this repository's code. |
| D-32 | Figures | Whether dataset frames may appear in published figures. |
| D-34 | RTMPose invalid keypoints in recogniser inputs | How RTMPose keypoints with score ≤ 0 (location set to −1 before rescaling) are represented in recogniser inputs. This is a separate preprocessing question, linked to D-07b but not part of its Option A/C choice. Undecided (Section 5.5.3). |

---

## 15. References

1. Zhang, W., Zhu, M., & Derpanis, K. G. (2013). From Actemes to Action: A Strongly-supervised Representation for Detailed Action Understanding. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, pp. 2248–2255.
2. Penn Action Dataset. Official dataset page: https://dreamdragon.github.io/PennAction
3. Jiang, T., Lu, P., Zhang, L., Ma, N., Han, R., Lyu, C., Li, Y., & Chen, K. (2023). RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose. *arXiv preprint* arXiv:2303.07399.
4. MMPose Contributors. (2020). OpenMMLab Pose Estimation Toolbox and Benchmark. https://github.com/open-mmlab/mmpose
5. Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., & Zitnick, C. L. (2014). Microsoft COCO: Common Objects in Context. In *European Conference on Computer Vision (ECCV)*, pp. 740–755.
6. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.

*Reference details should be checked against the original publications before the protocol is finalised.*

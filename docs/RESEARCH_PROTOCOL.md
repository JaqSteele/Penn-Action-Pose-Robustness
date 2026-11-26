# Research Protocol (Draft)

**Project:** Penn Action: Pose Estimation Robustness in Human Action Recognition
**Document status:** Draft v0.4 for review. Not final.
**Date of this revision:** 2026-11-26 (v0.1 drafted 2026-11-18)
**Project status at time of writing:** Environment and repository set up. The Penn Action dataset has been downloaded and is stored locally, outside the repository. Its annotation files and a small sample of training frames have been inspected. No poses have been estimated and no experiments have been run. This document contains **no experimental results**.

### Revision history

| Version | Date | Summary |
|---|---|---|
| v0.1 | 2026-11-18 | First draft. |
| v0.2 | 2026-11-20 | Targeted revision following independent methodological review. Changes: a primary model-selection strategy (C1 validation, hyperparameters frozen across conditions); a 12-joint primary representation with the head excluded; corruption residuals restricted to the fitting subset; confidence scores excluded from recogniser inputs; visibility handling clarified; repeated C4 corruption draws favoured; paired bootstrap procedure specified; H6 framed as exploratory; PCK definition clarified. The research question, scope and conditions C1–C5 are unchanged. |
| v0.3 | 2026-11-20 | Resolved D-33. After selection on C1 validation and freezing of all settings, final models are reinitialised and trained from scratch on the full official training partition (1,258 videos), with the frozen epoch count and no validation-based early stopping. The corruption model remains derived from the fitting subset only. Sections 4.1, 4.4, 4.5, 7.1–7.4, 8.1, 9.2, 9.3, 11.2, 12 and 14 updated accordingly. No other methodological decisions changed. **v0.3 amendments after internal review (same date):** §4.3 design-stage pose-error analysis restricted to the fitting subset (previously "training partition"); §4.4 Stage B adds that data-fitted preprocessing is refitted on the full official training partition and never on the test partition, with methods still open under D-12 (later Stage B items renumbered); §11.2 Stage A log label corrected to "fitting + validation subsets"; §12 limitation widened from the epoch count to all frozen hyperparameters. No decisions resolved or changed. |
| v0.4 | 2026-11-26 | Resolved D-09 (left/right convention). Inspection of Penn Action training annotations and frames showed that columns labelled LEFT hold the subject's anatomical RIGHT, and vice versa. The Penn Action → COCO mapping therefore swaps every left/right pair (Sections 6.2, 6.4, 14). The original annotation files are unchanged; the swap belongs in project mapping code. Project-status line updated (dataset now stored locally). No other methodological decisions changed; D-11 remains open. |

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
9. **Data-fitted preprocessing.** Any preprocessing step whose parameters are estimated from data (for example, feature standardisation statistics) is **refitted on the full official training partition** (all 1,258 videos) for the final models. It is **never** fitted using the test partition. The specific preprocessing methods remain open under `D-12`.
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
- **Unconfirmed [OPEN]:** bounding-box padding, aspect-ratio adjustment, and the handling of frames with a missing or degenerate box (`D-05`).

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

Which ground-truth joints are valid for error measurement depends on the visibility semantics of the annotations (Section 5.5, `D-07`).

Head-joint error is outside the primary analysis and may be reported only in the exploratory analysis described in Section 6.3.

### 5.5 Visibility and missing joints **[OPEN]**

Penn Action provides a per-joint visibility flag. It is **not** assumed that every joint flagged as invisible should simply be discarded from every part of the study. For example, an occluded joint may still carry an annotated position of useful but lower reliability. Before deciding, the annotation format and the exact meaning of the visibility flag (and of any coordinate placeholders) will be inspected on the downloaded data. The following questions will then be decided separately (`D-07`):

1. which joints are valid for **pose-error measurement** (Section 5.4);
2. which observations are used to **fit corruption distributions** (Section 8);
3. how missing or invisible coordinates are **represented in recogniser inputs** (Section 9).

**[ESTABLISHED]** Whatever rules are adopted, they are applied **consistently across all conditions**, and recorded before any test-set evaluation.

---

## 6. Mapping between Penn Action and COCO joints

### 6.1 Primary joint representation: 12 joints **[ESTABLISHED]**

RTMPose outputs 17 COCO keypoints, and Penn Action annotates 13 joints. All conditions use **one common 12-joint primary representation**, made up of the joints that correspond directly in both formats: the left and right **shoulders, elbows, wrists, hips, knees and ankles**.

- The **head is excluded** from the primary representation in **every** condition, including ground-truth conditions.
- This is a **deliberate representation choice**, not missing data. No reliable correspondence has been established between any COCO keypoint and Penn Action's head annotation, and none is assumed or invented.
- The same 12-joint representation, in the same joint order, is used for recogniser inputs, error measurement and corruption in every condition.

### 6.2 Mapping table

Penn Action joint indices below follow the dataset's documented ordering (1-based); the 0-based column is the corresponding column of the `x`/`y`/`visibility` arrays. The column order is consistent with the annotations, but it has not yet been formally verified (`D-11`, open).

**Left/right is swapped [ESTABLISHED, v0.4].** Penn Action columns labelled LEFT hold the subject's anatomical RIGHT, and columns labelled RIGHT hold the subject's anatomical LEFT (`D-09`, Section 6.4). Each Penn Action joint is therefore mapped to the COCO keypoint for the **opposite-named** side. COCO keypoints are taken to use the subject's anatomical left/right (see the assumption in Section 6.4).

| Penn Action label (1-based index) | Penn column (0-based) | Subject's anatomical side (D-09) | COCO keypoint (0-based index) | Status |
|---|---|---|---|---|
| Head (1) | 0 | — | None assumed | **Excluded from primary representation [ESTABLISHED].** Exploratory only (`D-08`) |
| Left shoulder (2) | 1 | Right | Right shoulder (6) | Primary; swap **[ESTABLISHED]**; column order subject to `D-11` |
| Right shoulder (3) | 2 | Left | Left shoulder (5) | Primary; swap **[ESTABLISHED]**; column order subject to `D-11` |
| Left elbow (4) | 3 | Right | Right elbow (8) | Primary; swap **[ESTABLISHED]**; column order subject to `D-11` |
| Right elbow (5) | 4 | Left | Left elbow (7) | Primary; swap **[ESTABLISHED]**; column order subject to `D-11` |
| Left wrist (6) | 5 | Right | Right wrist (10) | Primary; swap **[ESTABLISHED]**; column order subject to `D-11` |
| Right wrist (7) | 6 | Left | Left wrist (9) | Primary; swap **[ESTABLISHED]**; column order subject to `D-11` |
| Left hip (8) | 7 | Right | Right hip (12) | Primary; swap **[ESTABLISHED]**; column order subject to `D-11` |
| Right hip (9) | 8 | Left | Left hip (11) | Primary; swap **[ESTABLISHED]**; column order subject to `D-11` |
| Left knee (10) | 9 | Right | Right knee (14) | Primary; swap **[ESTABLISHED]**; column order subject to `D-11` |
| Right knee (11) | 10 | Left | Left knee (13) | Primary; swap **[ESTABLISHED]**; column order subject to `D-11` |
| Left ankle (12) | 11 | Right | Right ankle (16) | Primary; swap **[ESTABLISHED]**; column order subject to `D-11` |
| Right ankle (13) | 12 | Left | Left ankle (15) | Primary; swap **[ESTABLISHED]**; column order subject to `D-11` |

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

**[PROPOSED]** The same pose normalisation and temporal preprocessing are applied in every condition, and the corruption and normalisation steps are performed in a fixed, documented order (`D-12`).

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
2. Compute residuals (estimated minus ground truth) per joint and per frame, in a normalised coordinate frame (`D-06`). Which observations are eligible is decided under `D-07` (Section 5.5).
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
- Pose normalisation: for example centring on the hip midpoint and scaling by torso length or box size (`D-12`).
- Representation of missing or invisible joints in model input (`D-07`).
- Feature engineering for classical models, such as summary statistics over time, velocities or joint angles (`D-19`).
- Class-imbalance handling, such as class weighting (`D-24`).

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

- **Ground-truth bounding-box crops** remove detection error, so the results likely underestimate the degradation that a fully automatic pipeline would show.
- **Model selection on C1 only:** hyperparameters and training duration chosen on clean ground-truth data may not suit C3 and C5. Their primary-experiment performance may therefore understate what condition-specific tuning could achieve. The optional secondary analysis (Section 4.4) addresses this.
- **Ground-truth annotation noise:** Penn Action annotations are themselves imperfect, so "error" is measured relative to a noisy reference.
- **Invisible or occluded joints** may have unreliable ground-truth positions, which affects error measurement, corruption fitting and recogniser inputs (Section 5.5).
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
| D-05 | Cropping | Bounding-box padding, aspect-ratio handling, and frames with missing or degenerate boxes. |
| D-06 | Error normalisation and PCK | Coordinate frame and reference scale for residuals and error metrics; PCK reference scale *s* and threshold fraction(s) *α*. |
| D-07 | Visibility and missing joints | After inspecting the annotation format and visibility semantics: the joints valid for error measurement, the observations used for corruption fitting, and the representation of missing or invisible joints in recogniser inputs. Treatment must be consistent across conditions. |
| D-08 *(exploratory part)* | Exploratory head mapping | Whether a defensible COCO-to-head correspondence exists (assessed on the fitting subset only), and if so which one. Otherwise the head analysis is not performed. |
| D-11 | Joint ordering | Verify Penn Action's joint index ordering against the downloaded annotations. |
| D-12 | Normalisation and ordering | Pose normalisation method, and the order of the normalisation and corruption steps. |
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

---

## 15. References

1. Zhang, W., Zhu, M., & Derpanis, K. G. (2013). From Actemes to Action: A Strongly-supervised Representation for Detailed Action Understanding. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, pp. 2248–2255.
2. Penn Action Dataset. Official dataset page: https://dreamdragon.github.io/PennAction
3. Jiang, T., Lu, P., Zhang, L., Ma, N., Han, R., Lyu, C., Li, Y., & Chen, K. (2023). RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose. *arXiv preprint* arXiv:2303.07399.
4. MMPose Contributors. (2020). OpenMMLab Pose Estimation Toolbox and Benchmark. https://github.com/open-mmlab/mmpose
5. Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., & Zitnick, C. L. (2014). Microsoft COCO: Common Objects in Context. In *European Conference on Computer Vision (ECCV)*, pp. 740–755.
6. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.

*Reference details should be checked against the original publications before the protocol is finalised.*

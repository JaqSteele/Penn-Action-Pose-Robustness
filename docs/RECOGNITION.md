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

---

# Recogniser and experiment infrastructure (Batch D2)

Batch D2 implements the Release 1.0 recogniser, training, metrics, selection, bootstrap and experiment management (protocol v0.14, Sections 9.5, 9.5.8 and 10.7; D-36 to D-45). **Nothing in D2 has been run on Penn Action data:**
- the device qualification has not been run;
- Stage A and Stage B have not been run;
- no validation or test metric has been computed.

## Modules

| Module | Contents | Protocol |
|---|---|---|
| `src/penn_pose/model.py` | `TemporalCNN` (133,135 parameters); `build_model`, which initialises from the run seed's "init" stream | D-21, D-39, D-44 |
| `src/penn_pose/determinism.py` | deterministic configuration and checks; stream seeds; model-state SHA-256; environment record and device lock | D-36, D-44 |
| `src/penn_pose/training.py` | class weights; weighted loss; AdamW; batches; `train_model` with state hashes at checkpoints; inference-mode evaluation | D-24, D-38–D-40, D-44 |
| `src/penn_pose/metrics.py` | macro-F1 over all 15 classes (never-predicted classes score 0), accuracy, per-class F1, confusion matrix, majority baseline | §10.1, D-37 |
| `src/penn_pose/selection.py` | the D-22 plan and selection; ties are exact float64 equality | D-22, D-44 |
| `src/penn_pose/bootstrap.py` | paired, class-stratified bootstrap; percentile intervals with explicit linear interpolation; sample SD of the seed differences; the seed must always be passed explicitly (settings below) | D-26–D-28, D-45–D-47 |
| `src/penn_pose/runs.py` | write-then-rename run folders, run and stage manifests, records that are never overwritten, the Git commit gate | D-41, D-42 |
| `src/penn_pose/experiment.py` | qualification, Stage A C1, selection record, C2 reruns, C3, Stage B | §4.4, §9.5, D-36, D-41, D-43 |
| `scripts/run_release1.py` | command-line interface; every training command needs `--confirm` | — |
| `configs/recognition/release1.json` | the committed experiment configuration (checked against the code), including the bootstrap settings below | D-42, D-45–D-47 |

## Final-evaluation bootstrap settings (protocol v0.15)

These are the settings in `configs/recognition/release1.json` (`"bootstrap"`) and `bootstrap.py`, fixed in protocol v0.15 (Sections 9.5.9 and 10.7) after Stage B and before any test-partition output existed. A test checks this table against the configuration.

| Setting | Value | Decision |
|---|---|---|
| Resamples (B) | `10000` | D-26 |
| Seed | `20261128` | D-45 |
| Seed-variation standard deviation | sample SD, `ddof=1` | D-46 |
| Percentile interpolation | `method="linear"` | D-47 |

The bootstrap is paired and class-stratified, with the same resamples for every condition, seed and comparison; intervals are 95% (2.5th and 97.5th percentiles) and Bonferroni-adjusted 97.5% (1.25th and 98.75th). The bootstrap functions have no default seed: the final evaluation passes the configured seed explicitly.

## Random-number streams (D-44)

- **Stream seeds.** Each run seed (0, 1, 2) yields three stream seeds, derived with SHA-256 from the seed and the stream name ("init", "order", "dropout"). They don't depend on the condition or the hyperparameters.
- **Initialisation.** Weights are created on the CPU with PyTorch's default initialisation, inside `torch.random.fork_rng`, from the "init" stream.
- **Batch order.** It comes from a dedicated CPU generator seeded from the "order" stream.
- **Dropout.** It uses the global generator, seeded from the "dropout" stream at the start of training.
- **Evaluation.** It runs in inference mode, without gradients and inside `fork_rng`, so it can't change the training sequence.
- **Matched seeds.** C1 and C3 runs with the same seed therefore have identical initial states and batch orders (tested).

## Order of operations (one command each, each needing separate approval)

1. `qualify`:
   - runs two repetitions, each in a fresh process, of a 5-epoch run on ground-truth fitting inputs;
   - compares the state hashes and training-only metrics exactly, with no validation or test data;
   - tries the GPU first, then the CPU with one thread, and otherwise stops;
   - writes the full qualification run outside Git and the compact record `configs/recognition/release1-device.json`, **which must be committed before Stage A**.
2. `stage-a-c1`: 24 runs, with ground-truth validation macro-F1 and the state hash at epochs 5–150. It resumes safely: finished runs are verified (manifest and provenance) and skipped.
3. `select`: writes `configs/recognition/release1-selection.json` (the selection and freeze record) and `results/tables/release1_stage_a_c1_validation.csv`. **The record must be committed before steps 4–6** (D-41; enforced in code).
4. `stage-a-c2`: reruns the selected configuration to E for every seed. Each rerun must reproduce the recorded state hash and validation macro-F1 exactly before any estimated-pose evaluation.
5. `stage-a-c3`: trains on estimated fitting inputs for E epochs and evaluates only at E (D-43).
6. `stage-b`: trains the frozen C1 models (shared with C2) and C3 models on all 1,258 videos, computes no metric, and saves the model weights outside Git. `experiment.load_stage_b_models` reloads them and verifies every manifest and state hash, for the separate final evaluation in Batch E (not implemented in D2; it needs the test-partition inputs and D-01; the D-45 seed is fixed in protocol v0.15).

Every command after `qualify` loads the committed device record and refuses to run if the device, GPU name, PyTorch, CUDA or cuDNN version differs. Outputs go to `C:\datasets\penn-action\experiments\release1\`, with a SHA-256 manifest per run and per stage. Git receives only the configuration, the compact records in `configs/recognition/` (`release1-device.json`, `release1-selection.json`, `release1-stage-a-c2.json`, `release1-stage-a-c3.json`, `release1-stage-b.json`) and the aggregate table in `results/tables/`. Every writer refuses an output root inside the repository.

## Provenance and code identity

These checks implement D-36 (one qualified device and its software throughout), D-41 (exact reruns, and the selection record committed before any C2, C3 or Stage B step) and §9.5.7 (the code commit recorded before Stage B). They add no methodological decision.

**Protected scope.** The files that determine what an experiment computes (`runs.PROTECTED_PATHS`):

| Path | Why it is protected |
|---|---|
| `src/` | all implementation code (D1 preprocessing and D2) |
| `scripts/` | the command-line entry points |
| `configs/recognition/release1.json` | the experiment configuration (D-42) |
| `configs/inputs/release1-cache.json` | the recorded D1 input-cache hashes |
| `configs/splits/` | the D-02 fitting and validation lists |
| `requirements.txt` | the main-environment dependency specification |
| `docs/RESEARCH_PROTOCOL.md` | the authoritative research protocol (v0.14) that the experiment follows |

This list is canonical. Provenance must record exactly this list, in this order. A narrowed, extended, reordered or altered list is rejected, and every check recomputes identities from the canonical list rather than from recorded values.

Records (`configs/recognition/release1-*.json`), result tables, explanatory documentation (including this file) and tests are outside the scope. Committing them never changes the code identity, and they may make the rest of the working tree dirty. A committed protocol amendment *does* change the identity. Stage A, selection, C2, C3 and Stage B must therefore all run under the same protocol text.

**Code identity.** `runs.code_identity` refuses unless the protected scope is committed and clean. It reports any of these:
- a tracked modification;
- a staged change;
- an untracked file;
- an ignored file. Only `__pycache__` bytecode is allowed, because Python uses it only when the matching committed source file exists.

It returns HEAD and the SHA-256 of `git ls-tree -r` for the protected paths at HEAD. Every protected path must exist in that commit. Because the D2 implementation must be committed before this succeeds, no experiment step can run on uncommitted code.

**Required provenance.** `experiment.provenance` records:
- the protocol version;
- the environment (`determinism.environment_record`);
- the Git state;
- the code identity;
- the device-record hash;
- the split hashes;
- the input-record hash.

Every step (`stage_a_c1`, `select_and_record`, `stage_a_c2`, `stage_a_c3`, `stage_b`) requires it. `validate_provenance` refuses provenance in any of these cases:
- it is missing or incomplete;
- a locked environment field is missing, null or malformed. `device` must be a CPU or CUDA device matching `device_type`; `device_name`, `torch` and `cuda` must be non-empty strings; `cudnn` must be a positive integer;
- `device_record_sha256`, `inputs_record_sha256`, the two split hashes (exactly the keys `fitting` and `validation`) or the protected-tree hash is not a 64-character hexadecimal SHA-256, in either case;
- its protected paths are not the canonical list;
- its code identity doesn't describe the repository as it is now.

**Locked fields** (`check_provenance`):
- the protocol version, the device-record hash, the split hashes and the input-record hash;
- the environment fields in `determinism.LOCKED_FIELDS`: device, device type, device name, PyTorch, CUDA and cuDNN;
- the protected paths and protected-tree hash.

The NumPy, SciPy, scikit-learn and Python versions and the overall dirty flag are recorded but not locked.

**How it applies at each step:**

| Step | Rule |
|---|---|
| Stage A C1 | Before anything new is trained, every finished run (including resumed runs) must match the current provenance **with the same implementation commit**. Provenance is re-validated before each new run is written. A mismatching run stops the stage and is never rewritten. |
| Selection | Every Stage A run must match the current provenance with the same commit, so no commit may be made between Stage A and `select`. The selection record stores the Stage A implementation (`stage_a_implementation`: commit, protected paths, protected-tree hash). |
| C2, C3, Stage B | The current provenance must match the selection record's. The Stage A commit must be an ancestor of HEAD, and the protected-tree hash at that commit must equal the recorded one and the current one. Record commits (such as the committed selection record) are therefore allowed, while any committed, staged, uncommitted or untracked change in the protected scope is refused. Provenance is re-validated before each run is written. |


**Training-only metrics.** The per-epoch training loss is the epoch-level class-weighted mean of the per-video losses, sum(w_i l_i) / sum(w_i), so the smaller last batch counts by its weight. The training accuracy is the proportion of training videos whose training-mode prediction was correct during the epoch. These metrics are logged and compared in the qualification; selection never uses them.

## Known behaviour

On a machine with a visible CUDA GPU, the first `AdamW.step()` initialises a CUDA context, even when every tensor is on the CPU. The call chain is `torch/optim/adam.py` `step` → `Optimizer._accelerator_graph_capture_health_check` → `torch.accelerator.current_stream().is_capturing()`. This doesn't change CPU results. `CUBLAS_WORKSPACE_CONFIG` is set when `penn_pose.determinism` is imported, before `torch` is first imported (tested in a fresh interpreter). Each qualification repetition, including the CPU fallback, runs in a new Python process, which applies `determinism.configure` before training.

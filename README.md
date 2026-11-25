# Penn Action: Pose Estimation Robustness in Human Action Recognition

## Research question

How does pose-estimation error affect the performance of downstream human action recognition models?

## Project overview

This project investigates how errors in estimated human pose keypoints propagate into human action recognition.

Using the Penn Action dataset, the planned experiments compare action recognition performance using ground-truth poses, estimated poses, and controlled pose corruption.

The aim is to conduct a controlled empirical study rather than propose a new pose-estimation or action-recognition architecture.

## Planned methodology

The experimental design includes five conditions:

- C1: Ground-truth poses for training and testing.
- C2: Ground-truth poses for training and estimated poses for testing.
- C3: Estimated poses for training and testing.
- C4: Ground-truth poses for training and artificially corrupted testing poses.
- C5: Ground-truth poses with corruption augmentation for training, and estimated poses for testing.

Planned evaluation includes accuracy, macro-F1, confidence intervals, and analysis of errors across classes and keypoints.

## Dataset

Penn Action: https://dreamdragon.github.io/PennAction

The dataset is intended for research use. Dataset files and derived annotations will not be redistributed through this repository.

## Project status

**Setup in progress.**

The Python environment, GPU-enabled PyTorch installation, project directory structure, and Git repository have been established. Data preparation, pose estimation, and experiments have not yet been completed.

| Component | Status |
|---|---|
| Research protocol | Draft v0.3 ([`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md)) |
| Pose estimator (RTMPose-M) | Separate environment installed and smoke-tested. These tests verify software functionality only, **not** accuracy on Penn Action. See [`docs/POSE_ESTIMATION_SETUP.md`](docs/POSE_ESTIMATION_SETUP.md). |
| Estimator settings | Recorded in [`configs/pose/rtmpose_m.yaml`](configs/pose/rtmpose_m.yaml); not yet formally frozen |
| 17-to-12 joint mapping, pose extraction, experiments | Not started |

## Reproducibility

The main research environment's Python dependencies are recorded in `requirements.txt`, and its environment and hardware information in `environment-info.txt`.

The RTMPose pose estimator runs in a separate, pinned environment, because the official MMPose software is incompatible with the main environment's PyTorch and NumPy versions. Its lock file and setup notes are in [`environments/rtmpose/`](environments/rtmpose/README.md).

## Repository structure

- `configs/` - Experiment and estimator configuration files.
- `data/` - Dataset storage and processed data.
- `docs/` - Research protocol and setup documentation.
- `environments/` - Records of separate software environments (e.g. RTMPose).
- `experiments/` - Experiment scripts and records.
- `notebooks/` - Exploratory analysis.
- `src/` - Project source code.
- `results/` - Generated figures and tables.
- `checkpoints/` - Model checkpoints.
- `logs/` - Training and experiment logs.

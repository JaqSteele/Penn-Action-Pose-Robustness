# RTMPose estimator environment

This folder records the **separate Python environment** used to run the pretrained RTMPose-M pose estimator. The environment itself lives outside the repository at `C:\venvs\penn-action-rtmpose` and is not committed.

| File | Purpose |
|---|---|
| `README.md` | Why the environment exists, how it was built, how to verify and reproduce it |
| `requirements-rtmpose.lock.txt` | Exact package list (`pip freeze --all`) from the working environment |
| `environment-info-rtmpose.txt` | Verified Python, PyTorch, CUDA, cuDNN, GPU and driver details |

Model, checkpoint and inference details are in [`docs/POSE_ESTIMATION_SETUP.md`](../../docs/POSE_ESTIMATION_SETUP.md). Estimator settings are in [`configs/pose/rtmpose_m.yaml`](../../configs/pose/rtmpose_m.yaml).

## Why a separate environment?

The main research environment (`.venv`, described by the top-level `requirements.txt`) uses PyTorch 2.15 with CUDA 13.4 and NumPy 2.4. The official MMPose software cannot be installed there:

- **MMDetection caps MMCV.** RTMPose's backbone (CSPNeXt) is defined in MMDetection, so MMDetection is required. MMDetection 3.2.0 only accepts `mmcv < 2.2.0`. MMPose 1.3.2's own install file also requires `mmdet < 3.3.0`.
- **MMCV 2.1.0 has prebuilt Windows wheels only up to PyTorch 2.1.** These are on the official OpenMMLab wheel index. For any newer PyTorch it would have to be compiled from source.
- **The older stack needs NumPy 1.x.** Compiled packages from that period were built for NumPy 1.x.

So the estimator runs in its own pinned environment, and the two environments **never share a Python process**. The intended design is that the RTMPose environment only *extracts* poses and writes them to files. The main environment reads those files with NumPy and never imports MMPose. Once the estimator settings are formally frozen, this design will keep the estimator fixed and identical for every experimental condition. The settings are **not yet frozen**, and the pose-file format and extraction scripts are **not yet implemented**.

## Installed versions

| Component | Version |
|---|---|
| Python | 3.11.9 |
| PyTorch / torchvision | 2.1.2+cu121 / 0.16.2+cu121 (CUDA 12.1 runtime, cuDNN 8.8.1) |
| MMCV | 2.1.0 (prebuilt wheel with compiled ops) |
| MMEngine | 0.10.7 |
| MMDetection | 3.2.0 |
| MMPose | 1.3.2 |
| NumPy | 1.26.4 |
| OpenCV (opencv-python) | 4.11.0.86 |

The complete list is in `requirements-rtmpose.lock.txt`. `chumpy`, a declared MMPose dependency, is **intentionally absent**. It is distributed only as source code and has known build failures. It is not required for RTMPose inference: MMPose imported, loaded the model and ran RTMPose inference in this environment without it.

## How it was installed

The environment was built with the pinned procedure reviewed during development. The procedure is summarised below. **The lock file is the authoritative record of what is installed.**

1. Create a venv from the existing Python 3.11.9 interpreter at `C:\venvs\penn-action-rtmpose`.
2. Install with a **constraints file** that pins the critical versions. pip then fails loudly rather than silently replacing them:
   ```
   numpy==1.26.4
   torch==2.1.2
   torchvision==0.16.2
   mmcv==2.1.0
   mmengine==0.10.7
   mmdet==3.2.0
   opencv-python==4.11.0.86
   xtcocotools==1.14.3
   pycocotools==2.0.7
   ```
   OpenCV is pinned because `opencv-python` 4.12 and later require NumPy 2. pycocotools is pinned to a NumPy-1-era wheel.
3. Install in this order, using binary wheels only:
   1. NumPy.
   2. torch and torchvision, from `https://download.pytorch.org/whl/cu121`.
   3. MMCV, from `https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html`. PyPI has only source code for MMCV 2.1.0.
   4. MMEngine and MMDetection.
   5. MMPose, with `--no-deps`.
   6. The remaining MMPose runtime dependencies, without chumpy.

The setup folder `C:\venvs\penn-action-rtmpose-setup` (outside the repository) holds the constraints file, the downloaded checkpoint and the test images. None of these are committed.

## Activating and verifying on Windows

The safest way to use the environment is to **call its Python directly**, without activating it. That way nothing changes `PATH`, and the project's `.venv` cannot be affected:

```powershell
& 'C:\venvs\penn-action-rtmpose\Scripts\python.exe' -c "import torch, mmcv, mmengine, mmdet, mmpose, numpy; print(torch.__version__, torch.cuda.is_available(), mmcv.__version__, mmengine.__version__, mmdet.__version__, mmpose.__version__, numpy.__version__)"
```

Expected output: `2.1.2+cu121 True 2.1.0 0.10.7 3.2.0 1.3.2 1.26.4`

Check that MMCV's compiled extensions load. This is the exact import that fails on a broken install:

```powershell
& 'C:\venvs\penn-action-rtmpose\Scripts\python.exe' -c "from mmcv.ops import MultiScaleDeformableAttention; print('mmcv ops OK')"
```

Check package consistency:

```powershell
& 'C:\venvs\penn-action-rtmpose\Scripts\python.exe' -m pip check
```

The only expected message is: `mmpose 1.3.2 requires chumpy, which is not installed.`

If you prefer to activate it for an interactive session, run `C:\venvs\penn-action-rtmpose\Scripts\Activate.ps1`. This may need an execution-policy change in PowerShell, and you should run `deactivate` before going back to the main project.

## Reproducing the installation

These instructions have **not yet been tested** on a clean machine. The lock file has not yet been used to rebuild the environment from scratch.

### Prerequisites

- 64-bit Windows. The original environment was built on Windows 11; other versions are untested.
- **Python 3.11** (64-bit). The wheels are built for Python 3.11; the original environment used **3.11.9**.
- An NVIDIA GPU with a driver that supports **CUDA 12.1**. No separate CUDA toolkit is needed, because the PyTorch wheels bundle the CUDA runtime. The original machine used an RTX 3060 Ti with driver 617.42.
- Internet access for steps 2 and 3. Expect several gigabytes of downloads, mostly the PyTorch wheel.
- A PowerShell window opened **in the repository root**, because the commands below use paths relative to it.

### 1. Create the environment

Replace `<user>` with your Windows user name, or point to wherever Python 3.11 is installed:

```powershell
& 'C:\Users\<user>\AppData\Local\Programs\Python\Python311\python.exe' -m venv C:\venvs\penn-action-rtmpose
```

### 2. Install the exact package versions (online)

```powershell
& 'C:\venvs\penn-action-rtmpose\Scripts\python.exe' -m pip install --no-deps --only-binary=:all: -r environments\rtmpose\requirements-rtmpose.lock.txt --extra-index-url https://download.pytorch.org/whl/cu121 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
```

Why each option is there:

- `--no-deps`: the lock file already lists every package. It also stops pip from trying to install `chumpy`.
- `--only-binary=:all:`: forbids source builds, so a missing wheel fails loudly.
- `--extra-index-url`: supplies the `+cu121` PyTorch wheels.
- `-f`: supplies the prebuilt MMCV 2.1.0 Windows wheel.

Then run the verification commands in the previous section. `pip check` should report only the expected chumpy message.

### 3. Download and verify the checkpoint (online, then offline)

The checkpoint is **not** stored in this repository. Download it once, outside the repository, then check its SHA-256 against the value recorded in `docs/POSE_ESTIMATION_SETUP.md`. Run these three commands one at a time.

Create the folder for the checkpoint. `| Out-Null` only hides this command's output:

```powershell
New-Item -ItemType Directory -Force -Path 'C:\venvs\penn-action-rtmpose-setup\checkpoints' | Out-Null
```

Download the checkpoint (online):

```powershell
Invoke-WebRequest -Uri 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth' -OutFile 'C:\venvs\penn-action-rtmpose-setup\checkpoints\rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth'
```

Calculate its SHA-256 (offline):

```powershell
(Get-FileHash -Algorithm SHA256 -LiteralPath 'C:\venvs\penn-action-rtmpose-setup\checkpoints\rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth').Hash
```

Expected SHA-256 (PowerShell prints it in upper case): `929F00F2D84764AA0297DECF13A84556642E43CBF5FCA8CFB16E7326D92DFD97`. The file should be 54,535,816 bytes. If the hash differs, do not use the file.

### 4. Locate the model configuration (offline)

No separate config download is needed. The official RTMPose-M config ships inside the installed MMPose package:

```
C:\venvs\penn-action-rtmpose\Lib\site-packages\mmpose\.mim\configs\body_2d_keypoint\rtmpose\coco\rtmpose-m_8xb256-420e_coco-256x192.py
```

Its SHA-256 should be `f27da50a81c3d3b5c426262118b9a030d15c9dbddccda7100b8cc0bcd07f6fff`, matching the copy used during setup. Its base file `_base_\default_runtime.py` is in the same `configs` folder. When loading the model with `mmpose.apis.init_model`, do **not** pass `cfg_options`; otherwise the backbone's separate pretrained-weights download is no longer disabled.

## Removing the environment

The environment is a self-contained folder. Deleting it does not affect the project, its `.venv`, or the base Python installation:

```powershell
Remove-Item -LiteralPath 'C:\venvs\penn-action-rtmpose' -Recurse -Force
```

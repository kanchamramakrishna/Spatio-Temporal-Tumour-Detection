# Source Code Guide

This guide describes the Python modules in `src/` and how they fit together. The repository has two model workflows:

1. **Longitudinal future scan prediction:** take two consecutive timepoints (scan + mask for each) as input and predict the future scan volume at the next timepoint.
2. **Tumor detection:** classify one 3D scan as tumor-positive or healthy.

The Streamlit interface can combine the detector and segmentation model, but the models are prepared and trained independently.

## Workflow

The high-level project diagram is available at [workflow_stpa.png](../workflow_stpa.png). The executable workflows represented by the source files are:

### A. Longitudinal future scan prediction

```text
Raw tumor scans and masks (.nii/.nii.gz)
              |
              v
preprocess.py
  - resize and normalize scans
  - resize and binarize masks
  - save scan.npy and mask.npy by patient/timepoint
              |
              v
dataset_loader.py
  - form consecutive triplets (T1, T2, T3)
  - input: [scan_T1, mask_T1, scan_T2, mask_T2]
  - target: future scan_T3
              |
              v
train.py --> model.py (3D U-Net) --> tumor_model.pth
              |
       +------+--------------------------+
       |                                 |
       v                                 v
predict.py                         evaluate.py
single-sample prediction           batch MAE/MSE/PSNR evaluation
and visualization                  and evaluation_results.csv
       |
       v
app.py / app_sample.py
interactive diagnosis and future-scan prediction UI
```

This track requires patients with at least 3 consecutive timepoints. The model predicts a future scan volume (intensity regression), not a future segmentation mask. Retraining is required after this change.

### B. Tumor-presence classification

```text
Raw tumor scans + healthy scans
              |
              v
preprocess_detector.py
  - resize and normalize one scan per sample
  - write scan arrays and labels.npy
              |
              v
train_detector.py --> detector.py (3D CNN) --> detector_model.pth
                                              |
                                              v
                             app.py / app_sample.py
                             detection and Grad-CAM view
```

### C. Longitudinal volume analysis and manual viewing

```text
Raw patient folders with scans/masks at each timepoint
              |
       +------+-------------------------------+
       |                                      |
       v                                      v
analysis.py / main.py / volume_analysis.py   viewer_volume.py
volume statistics and growth plots           interactive slice viewer
```

`masking.py` is a small standalone example that overlays a mask on a scan. These analysis/viewer scripts are independent of model training.

## Files and responsibilities

| File | Responsibility |
|---|---|
| [`analysis.py`](analysis.py) | Console-based patient selection for raw longitudinal data. Computes mask volumes from NIfTI voxel spacing, plots volume over time, and offers a slice-by-slice scan viewer with a mask overlay. |
| [`app.py`](app.py) | Main Streamlit dashboard. Displays processed patient scans, tumor volumes, longitudinal trends, and MRI slices. Its diagnosis mode runs the detector, creates a Grad-CAM heatmap, and—when tumor is detected—uses two input timepoints (`scan+mask` for T1 and T2) to predict the future scan (T3). |
| [`app_sample.py`](app_sample.py) | Copy of `app.py` providing the same Streamlit interface; use `app.py` as the primary entry point unless a separate sample copy is specifically needed. |
| [`dataset_loader.py`](dataset_loader.py) | PyTorch dataset for longitudinal training. Finds consecutive triplets `(T1, T2, T3)` and returns `[scan_T1, mask_T1, scan_T2, mask_T2]` as input with future `scan_T3` as the target. |
| [`detector.py`](detector.py) | Defines `TumorDetector`, a 3D convolutional binary classifier that outputs a tumor-presence probability for one scan. |
| [`evaluate.py`](evaluate.py) | Loads `tumor_model.pth`, predicts every valid `(T1, T2) -> T3` sample in the processed dataset, computes MAE/MSE/PSNR against the actual future scan, and writes `evaluation_results.csv`. |
| [`gradcam.py`](gradcam.py) | Implements 3D Grad-CAM using model activations and gradients to produce a normalized attention heatmap. The Streamlit UI imports this version. |
| [`gradcam_plus.py`](gradcam_plus.py) | Alternate Grad-CAM implementation that additionally smooths the heatmap with a Gaussian filter. It is not the version imported by the current Streamlit UI. |
| [`main.py`](main.py) | Console-based longitudinal patient tracker. Reports mask volumes and plots growth, then optionally opens axial, sagittal, and coronal views. It does not train a model. |
| [`masking.py`](masking.py) | Standalone demonstration that loads a configured NIfTI scan and mask and plots a middle-slice overlay. Update its example paths before use. |
| [`model.py`](model.py) | Defines the 3D U-Net (`UNet3D`) used for volumetric prediction. For future-scan prediction it is configured with 4 input channels (two timepoints Ã— scan+mask) and 1 output channel (predicted future scan). |
| [`predict.py`](predict.py) | Loads trained weights, lets the user choose a patient and an input pair, predicts the future CT/MRI scan from two timepoints, reports MAE/MSE/PSNR against the actual future scan, and displays a comparison. |
| [`preprocess.py`](preprocess.py) | Prepares longitudinal data from raw NIfTI files. Resizes scans and masks to `(64, 64, 32)`, normalizes scans, binarizes masks, and saves `scan.npy`/`mask.npy` under the processed patient/timepoint folders. Also adds a limited set of healthy scans with zero masks. |
| [`preprocess_detector.py`](preprocess_detector.py) | Prepares independent scan-level examples for binary classification, saving tumor/healthy scan arrays and `labels.npy` under `processed_detector`. **It removes and recreates its configured output directory each time it runs.** |
| [`train.py`](train.py) | Trains the 3D U-Net on `(T1, T2) -> T3` samples with L1 loss to predict the future scan volume, then saves weights to `tumor_model.pth`. |
| [`train_detector.py`](train_detector.py) | Defines a dataset for `processed_detector`, trains the binary classifier with binary cross-entropy, and saves weights to `detector_model.pth`. |
| [`viewer_volume.py`](viewer_volume.py) | Opens a configured NIfTI volume in a Matplotlib window and allows the user to browse slices with a slider. The current example path points to a tumor-mask volume. |
| [`volume_analysis.py`](volume_analysis.py) | Standalone utility that calculates tumor volumes from two configured NIfTI masks, prints the change, and plots a two-timepoint comparison. |

## Expected data and generated files

The scripts currently use paths configured in their source files (many default to an `E:\EDI\tumor_growth_project\data\...` directory). Change those constants to match the local dataset before running.

The longitudinal preprocessing and training scripts expect data organized approximately as:

```text
data/
â”œâ”€â”€ raw/
â”‚   â”œâ”€â”€ MU-Glioma-Post/
â”‚   â”‚   â””â”€â”€ PatientID_.../
â”‚   â”‚       â””â”€â”€ Timepoint_.../
â”‚   â”‚           â”œâ”€â”€ ...brain_t1c....nii[.gz]
â”‚   â”‚           â””â”€â”€ ...tumorMask....nii[.gz]
â”‚   â””â”€â”€ Healthy/
â””â”€â”€ processed/
    â””â”€â”€ PatientID_.../
        â””â”€â”€ Timepoint_.../
            â”œâ”€â”€ scan.npy
            â””â”€â”€ mask.npy
```

The detector workflow uses a separate `processed_detector/` directory containing per-scan `.npy` files and `labels.npy`. Model weights and the evaluation CSV are written relative to the process's current working directory unless their configured paths are changed.

Install the base project dependencies from the repository root with:

```bash
pip install -r requirements.txt
```

`evaluate.py` imports `tqdm` for its progress bar, but `tqdm` is not currently listed in `requirements.txt`; install it separately if needed:

```bash
pip install tqdm
```

## Run commands

Run commands from the repository root, after updating data paths and ensuring the required input files exist.

```bash
# Prepare and train the longitudinal model
python src/preprocess.py
python src/train.py

# Prepare and train the detector
python src/preprocess_detector.py
python src/train_detector.py

# Launch the dashboard
streamlit run src/app.py

# Other entry points
python src/predict.py
python src/evaluate.py
python src/main.py
```

`preprocess.py` must be run before `train.py`; `preprocess_detector.py` must be run before `train_detector.py`. The dashboard expects processed longitudinal data. Its AI diagnosis mode also needs `detector_model.pth` and a `tumor_model.pth` trained with the updated two-input future-scan pipeline (and patients with >= 3 timepoints) for the full flow.

## Implementation notes

- The longitudinal predictor now takes **two input timepoints** (`scan` + `mask` for T1 and T2, 4 channels total) and predicts the **future scan volume** at T3 (1 output channel, intensity regression trained with L1 loss). This replaces the previous one-timepoint future-mask segmentation setup.
- `dataset_loader.py`, `predict.py`, `evaluate.py`, and the app predictor path permute saved arrays from `(H, W, D)` to `(D, H, W)` for model inference/training consistency.
- Training/evaluation samples require at least 3 consecutive timepoints per patient; healthy examples with a single timepoint do not form longitudinal samples.
- `tumor_model.pth` weights from the old mask-forecasting model are incompatible with the new 4-channel future-scan model and must be retrained.
- The volume calculation in the raw-NIfTI analysis scripts uses voxel spacing from the NIfTI header. The dashboard estimates volume from the resized `.npy` mask using a fixed voxel volume, so its values are approximate and may not match the raw-image measurements.
- This is a research/educational codebase, not a medical diagnostic system. Model outputs require qualified clinical review and should not be used as a substitute for it.

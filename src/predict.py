import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import UNet3D

# --- CONFIGURATION ---
MODEL_PATH = "tumor_model.pth"
DATA_DIR = r"E:\EDI\tumor_growth_project\data\processed"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_INPUT_TIMEPOINTS = 2
IN_CHANNELS = NUM_INPUT_TIMEPOINTS * 2  # scan + mask per timepoint
OUT_CHANNELS = 1  # predicted future CT scan


def _to_dhw(arr):
    a = np.asarray(arr)
    if a.ndim == 4:
        a = a.squeeze()
    if a.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {a.shape}")
    # Saved as (H, W, D); model/training use (D, H, W)
    return np.transpose(a, (2, 0, 1))


def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    model = UNet3D(n_channels=IN_CHANNELS, n_classes=OUT_CHANNELS).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print("✅ Model loaded successfully.")
        return model
    except FileNotFoundError:
        print("❌ Model file not found. Run 'train.py' after retraining for future-scan prediction.")
        raise


def get_patient_sample(patient_id, tp_idx=0):
    """
    Loads two consecutive input timepoints and the future target scan.
    Returns: input_tensor (1,C,D,H,W), future_scan (D,H,W), input_scans_dhw list, input_tps, target_tp
    """
    p_dir = os.path.join(DATA_DIR, patient_id)
    if not os.path.exists(p_dir):
        raise FileNotFoundError(f"Patient {patient_id} not found in processed data.")

    timepoints = sorted(
        [d for d in os.listdir(p_dir) if d.startswith("Timepoint_") and os.path.isdir(os.path.join(p_dir, d))],
        key=lambda x: int(x.split("_")[-1]),
    )

    needed = tp_idx + NUM_INPUT_TIMEPOINTS + 1
    if len(timepoints) < needed:
        raise ValueError(
            f"Patient needs at least {NUM_INPUT_TIMEPOINTS + 1} timepoints for two-input future prediction. "
            f"Found {len(timepoints)}."
        )

    input_tps = timepoints[tp_idx:tp_idx + NUM_INPUT_TIMEPOINTS]
    target_tp = timepoints[tp_idx + NUM_INPUT_TIMEPOINTS]
    print(f"Predicting future scan: {input_tps[0]} + {input_tps[1]} -> {target_tp}")

    channels = []
    input_scans_dhw = []
    for tp in input_tps:
        scan = _to_dhw(np.load(os.path.join(p_dir, tp, "scan.npy")))
        mask = _to_dhw(np.load(os.path.join(p_dir, tp, "mask.npy")))
        input_scans_dhw.append(scan)
        channels.extend([
            torch.from_numpy(scan).float(),
            torch.from_numpy(mask).float(),
        ])

    future_scan = _to_dhw(np.load(os.path.join(p_dir, target_tp, "scan.npy")))
    input_tensor = torch.stack(channels, dim=0).unsqueeze(0)
    return input_tensor, future_scan, input_scans_dhw, input_tps, target_tp


def calculate_metrics(pred, target):
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    mae = float(np.mean(np.abs(pred - target)))
    mse = float(np.mean((pred - target) ** 2))
    psnr = float(20 * np.log10(1.0 / np.sqrt(mse + 1e-8)))
    return {"mae": mae, "mse": mse, "psnr": psnr}


def visualize_prediction(input_scans_dhw, true_scan, pred_scan, patient_id, input_tps, target_tp, metrics):
    d_idx = true_scan.shape[0] // 2

    panels = [
        (np.rot90(input_scans_dhw[0][d_idx]), f"Input Scan ({input_tps[0]})"),
        (np.rot90(input_scans_dhw[1][d_idx]), f"Input Scan ({input_tps[1]})"),
        (np.rot90(true_scan[d_idx]), f"Actual Future Scan ({target_tp})"),
        (np.rot90(pred_scan[d_idx]), f"Predicted Future Scan (MAE {metrics['mae']:.4f})"),
    ]

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    for i, (img, title) in enumerate(panels):
        ax[i].imshow(img, cmap="gray", vmin=0, vmax=1)
        ax[i].set_title(title)
        ax[i].axis("off")

    plt.suptitle(f"Future CT Scan Prediction: {patient_id}", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model = load_model()

    patients = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    print("\nAvailable Patients:")
    for i, p in enumerate(patients[:20]):
        print(f" [{i}] {p}")

    try:
        choice = int(input("Select patient index: "))
        patient_id = patients[choice]

        p_dir = os.path.join(DATA_DIR, patient_id)
        timepoints = sorted(
            [d for d in os.listdir(p_dir) if d.startswith("Timepoint_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        max_start = len(timepoints) - (NUM_INPUT_TIMEPOINTS + 1)
        if max_start < 0:
            raise ValueError("Selected patient does not have enough timepoints (need >= 3).")

        tp_idx = 0
        if max_start > 0:
            raw = input(f"Select input pair start index [0-{max_start}] (default 0): ").strip()
            if raw:
                tp_idx = int(raw)
        if tp_idx < 0 or tp_idx > max_start:
            raise ValueError(f"Input pair start index must be between 0 and {max_start}.")

        input_tensor, true_scan, input_scans_dhw, input_tps, target_tp = get_patient_sample(patient_id, tp_idx=tp_idx)
        input_tensor = input_tensor.to(DEVICE)

        print("Running neural network...")
        with torch.no_grad():
            pred_scan = model(input_tensor).clamp(0.0, 1.0).cpu().numpy()[0, 0]

        metrics = calculate_metrics(pred_scan, true_scan)
        print("\n✅ Prediction complete.")
        print(f"MAE:  {metrics['mae']:.6f}")
        print(f"MSE:  {metrics['mse']:.6f}")
        print(f"PSNR: {metrics['psnr']:.2f} dB")

        visualize_prediction(input_scans_dhw, true_scan, pred_scan, patient_id, input_tps, target_tp, metrics)

    except Exception as e:
        print(f"Error: {e}")
import os
import csv
import torch
import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    def tqdm(x, **kwargs):
        return x

from model import UNet3D

# --- CONFIGURATION ---
MODEL_PATH = "tumor_model.pth"
DATA_DIR = r"E:\EDI\tumor_growth_project\data\processed"
OUTPUT_CSV = "evaluation_results.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_INPUT_TIMEPOINTS = 2
IN_CHANNELS = NUM_INPUT_TIMEPOINTS * 2
OUT_CHANNELS = 1


def _to_dhw(arr):
    a = np.asarray(arr)
    if a.ndim == 4:
        a = a.squeeze()
    if a.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {a.shape}")
    return np.transpose(a, (2, 0, 1))


def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    model = UNet3D(n_channels=IN_CHANNELS, n_classes=OUT_CHANNELS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def calculate_metrics(pred, target):
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    mae = float(np.mean(np.abs(pred - target)))
    mse = float(np.mean((pred - target) ** 2))
    psnr = float(20 * np.log10(1.0 / np.sqrt(mse + 1e-8)))
    return mae, mse, psnr


def evaluate_all():
    model = load_model()
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Processed data directory not found: {DATA_DIR}")

    patients = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    results = []
    print(f"\nStarting batch evaluation for future-scan prediction on {len(patients)} patients...")

    for patient_id in tqdm(patients):
        p_dir = os.path.join(DATA_DIR, patient_id)
        timepoints = sorted(
            [d for d in os.listdir(p_dir) if d.startswith("Timepoint_") and os.path.isdir(os.path.join(p_dir, d))],
            key=lambda x: int(x.split("_")[-1]),
        )

        if len(timepoints) < NUM_INPUT_TIMEPOINTS + 1:
            continue

        for i in range(len(timepoints) - NUM_INPUT_TIMEPOINTS):
            input_tps = timepoints[i:i + NUM_INPUT_TIMEPOINTS]
            target_tp = timepoints[i + NUM_INPUT_TIMEPOINTS]
            try:
                channels = []
                for tp in input_tps:
                    scan = _to_dhw(np.load(os.path.join(p_dir, tp, "scan.npy")))
                    mask = _to_dhw(np.load(os.path.join(p_dir, tp, "mask.npy")))
                    channels.extend([torch.from_numpy(scan).float(), torch.from_numpy(mask).float()])

                input_tensor = torch.stack(channels, dim=0).unsqueeze(0).to(DEVICE)
                target_scan = _to_dhw(np.load(os.path.join(p_dir, target_tp, "scan.npy")))

                with torch.no_grad():
                    pred_scan = model(input_tensor).clamp(0.0, 1.0).cpu().numpy()[0, 0]

                mae, mse, psnr = calculate_metrics(pred_scan, target_scan)
                results.append({
                    "Patient": patient_id,
                    "Inputs": f"{input_tps[0]} + {input_tps[1]}",
                    "Target": target_tp,
                    "MAE": mae,
                    "MSE": mse,
                    "PSNR_dB": psnr,
                })
            except Exception as e:
                print(f"Error processing {patient_id} ({input_tps} -> {target_tp}): {e}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Patient", "Inputs", "Target", "MAE", "MSE", "PSNR_dB"])
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 36)
    print("            FINAL RESULTS            ")
    print("=" * 36)
    if results:
        maes = [r["MAE"] for r in results]
        mses = [r["MSE"] for r in results]
        psnrs = [r["PSNR_dB"] for r in results]
        print(f"Total samples tested: {len(results)}")
        print(f"Average MAE:  {float(np.mean(maes)):.6f}")
        print(f"Average MSE:  {float(np.mean(mses)):.6f}")
        print(f"Average PSNR: {float(np.mean(psnrs)):.2f} dB")
        print(f"Best MAE:     {float(np.min(maes)):.6f}")
        print(f"Worst MAE:    {float(np.max(maes)):.6f}")
    else:
        print("No evaluable samples found. Need patients with >= 3 consecutive timepoints.")
    print("=" * 36)
    print(f"Detailed report saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    evaluate_all()
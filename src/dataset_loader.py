import os
import numpy as np
import torch
from torch.utils.data import Dataset

# --- CONFIGURATION ---
PROCESSED_DIR = r"E:\EDI\tumor_growth_project\data\processed"


def _to_tensor_dhw(arr):
    """Convert saved array (H, W, D) to tensor (D, H, W)."""
    tensor = torch.from_numpy(arr).float()
    if tensor.ndim == 3:
        tensor = tensor.permute(2, 0, 1)
    elif tensor.ndim == 4:
        # Some preprocessing paths may store an extra channel/singleton dimension.
        tensor = tensor.squeeze().permute(2, 0, 1)
    return tensor


class TumorGrowthDataset(Dataset):
    """
    Dataset for future CT scan prediction from two input timepoints.

    Each sample uses consecutive timepoints (T1, T2) as input and predicts the
    scan volume at T3:
      - Input:  [scan_T1, mask_T1, scan_T2, mask_T2] -> (4, D, H, W)
      - Target: [scan_T3]                             -> (1, D, H, W)
    """

    def __init__(self, data_dir=PROCESSED_DIR, num_input_timepoints=2):
        if num_input_timepoints != 2:
            raise ValueError("This dataset currently supports exactly 2 input timepoints.")
        self.data_dir = data_dir
        self.num_input_timepoints = num_input_timepoints
        self.samples = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Processed data directory not found: {self.data_dir}")

        patients = sorted([p for p in os.listdir(self.data_dir) if not p.startswith(".") and os.path.isdir(os.path.join(self.data_dir, p))])
        valid_samples = 0

        for patient in patients:
            patient_dir = os.path.join(self.data_dir, patient)
            timepoints = sorted(
                [d for d in os.listdir(patient_dir) if d.startswith("Timepoint_") and os.path.isdir(os.path.join(patient_dir, d))],
                key=lambda x: int(x.split("_")[-1]),
            )

            # Need N input timepoints plus 1 future target.
            if len(timepoints) < self.num_input_timepoints + 1:
                continue

            for i in range(len(timepoints) - self.num_input_timepoints):
                input_tps = timepoints[i:i + self.num_input_timepoints]
                target_tp = timepoints[i + self.num_input_timepoints]

                paths = {"patient_id": patient, "input_timepoints": input_tps, "target_timepoint": target_tp}
                ok = True
                for j, tp in enumerate(input_tps):
                    scan_path = os.path.join(patient_dir, tp, "scan.npy")
                    mask_path = os.path.join(patient_dir, tp, "mask.npy")
                    if not (os.path.exists(scan_path) and os.path.exists(mask_path)):
                        ok = False
                        break
                    paths[f"scan_{j}"] = scan_path
                    paths[f"mask_{j}"] = mask_path

                target_scan_path = os.path.join(patient_dir, target_tp, "scan.npy")
                if not ok or not os.path.exists(target_scan_path):
                    continue

                paths["target_scan"] = target_scan_path
                self.samples.append(paths)
                valid_samples += 1

        print(
            f"Dataset initialized: {valid_samples} samples found "
            f"({self.num_input_timepoints} input timepoints -> predict future scan)."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            channels = []
            for j in range(self.num_input_timepoints):
                scan = _to_tensor_dhw(np.load(sample[f"scan_{j}"]))
                mask = _to_tensor_dhw(np.load(sample[f"mask_{j}"]))
                channels.extend([scan, mask])

            input_tensor = torch.stack(channels, dim=0)
            target_scan = _to_tensor_dhw(np.load(sample["target_scan"])).unsqueeze(0)
            return input_tensor, target_scan
        except Exception as e:
            print(f"⚠️ Error loading sample {idx} ({sample.get('patient_id', '?')}): {e}. Returning zeros.")
            c = self.num_input_timepoints * 2
            return torch.zeros(c, 32, 64, 64), torch.zeros(1, 32, 64, 64)


if __name__ == "__main__":
    ds = TumorGrowthDataset()
    if len(ds) > 0:
        x, y = ds[0]
        print(f"Sample loaded. Input: {x.shape}, Target: {y.shape}")
    else:
        print("No samples found. Ensure patients have >= 3 consecutive timepoints.")
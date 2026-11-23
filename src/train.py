import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset_loader import TumorGrowthDataset
from model import UNet3D

# --- CONFIGURATION ---
BATCH_SIZE = 2
LEARNING_RATE = 1e-3
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "tumor_model.pth"
NUM_INPUT_TIMEPOINTS = 2
IN_CHANNELS = NUM_INPUT_TIMEPOINTS * 2  # scan + mask per timepoint
OUT_CHANNELS = 1  # predicted future CT scan intensity

# Number of CPU workers for data loading. Use 0 on Windows if multiprocessing fails.
NUM_WORKERS = 0


def train_model():
    print(f"--- Starting Future-Scan Training on {DEVICE} ---")
    print(f"Input: {NUM_INPUT_TIMEPOINTS} timepoints ({IN_CHANNELS} channels), Target: future scan")

    try:
        dataset = TumorGrowthDataset(num_input_timepoints=NUM_INPUT_TIMEPOINTS)
        if len(dataset) == 0:
            print("❌ No training samples found. Need patients with >= 3 consecutive timepoints.")
            return
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=(DEVICE == "cuda"),
        )
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    model = UNet3D(n_channels=IN_CHANNELS, n_classes=OUT_CHANNELS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()  # MAE is robust for intensity regression
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"Training for {EPOCHS} epochs (L1 loss, target = future scan)...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                predictions = model(inputs)
                loss = criterion(predictions, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            print(f"   E{epoch+1} | Batch {batch_idx+1}/{len(loader)} | L1: {loss.item():.4f}")

        elapsed = time.time() - start_time
        avg_loss = epoch_loss / max(len(loader), 1)
        print(f"Epoch {epoch+1}/{EPOCHS} Done | Avg L1: {avg_loss:.4f} | Total Time: {elapsed/60:.1f} min")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n✅ Training complete. Saved to '{SAVE_PATH}'")
    print("Note: Retrain is required; old mask-prediction weights are not compatible.")


if __name__ == "__main__":
    train_model()
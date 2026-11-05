
import sys
from pathlib import Path

# ==========================================
# Project path
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import time
import torch
import torch.nn.functional as F
from torch.optim import Adam

from models.acu_net import ACUNet
from data.dataloader import get_dataloaders


# ==========================================
# 1. Configuration
# ==========================================

DATASET_PATH = (
    r"E:\3.2\Brain Tumor Research\BrainTumor_UNet"
    r"\Datasets\BraTS2018"
)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 1
LEARNING_RATE = 0.0001
NUM_EPOCHS = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

RESUME = False


# ==========================================
# 2. Dice Loss
# ==========================================

def dice_loss(logits, targets, smooth=1.0):

    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.view(-1)
    targets = targets.view(-1)

    intersection = (probabilities * targets).sum()

    dice = (
        (2.0 * intersection + smooth)
        /
        (probabilities.sum() + targets.sum() + smooth)
    )

    return 1.0 - dice


# ==========================================
# 3. Combined Loss
# ==========================================

def combined_loss(logits, targets):

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets
    )

    dice = dice_loss(logits, targets)

    return bce + dice


# ==========================================
# 4. Calculate Dice Score
# ==========================================

def calculate_dice(logits, targets):

    predictions = torch.sigmoid(logits) > 0.5

    predictions = predictions.float()

    intersection = (predictions * targets).sum()

    denominator = predictions.sum() + targets.sum()

    dice = (2.0 * intersection + 1e-7) / (
        denominator + 1e-7
    )

    return intersection.item(), denominator.item()


# ==========================================
# 5. Training Function
# ==========================================

def train_one_epoch(model, loader, optimizer, epoch):

    model.train()

    total_loss = 0.0

    start_time = time.perf_counter()

    for batch_idx, (images, masks) in enumerate(loader):

        images = images.to(DEVICE)
        masks = masks.unsqueeze(1).to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)

        loss = combined_loss(logits, masks)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 500 == 0:

            print(
                f"Epoch {epoch} | "
                f"Batch {batch_idx + 1}/{len(loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    average_loss = total_loss / len(loader)

    return average_loss, elapsed


# ==========================================
# 6. Validation Function
# ==========================================

def validate(model, loader):

    model.eval()

    total_loss = 0.0

    total_intersection = 0.0
    total_denominator = 0.0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(DEVICE)
            masks = masks.unsqueeze(1).to(DEVICE)

            logits = model(images)

            loss = combined_loss(logits, masks)

            total_loss += loss.item()

            intersection, denominator = calculate_dice(
                logits,
                masks
            )

            total_intersection += intersection
            total_denominator += denominator

    average_loss = total_loss / len(loader)

    validation_dice = (
        (2.0 * total_intersection + 1e-7)
        /
        (total_denominator + 1e-7)
    )

    return average_loss, validation_dice


# ==========================================
# 7. Checkpoint Saving
# ==========================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    validation_dice,
    best_dice
):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_dice": validation_dice,
        "best_dice": best_dice
    }

    torch.save(
        checkpoint,
        CHECKPOINT_DIR / "latest_checkpoint.pth"
    )


# ==========================================
# 8. Main Function
# ==========================================

def main():

    print("Starting ACU-Net full training")
    print("================================")

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":

        print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------------------
    # Load datasets
    # --------------------------------------

    print("\nLoading datasets...")

    train_loader, val_loader = get_dataloaders(
        DATASET_PATH,
        batch_size=BATCH_SIZE
    )

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    # --------------------------------------
    # Initialize model
    # --------------------------------------

    print("\nInitializing ACU-Net...")

    model = ACUNet().to(DEVICE)

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_dice = 0.0
    start_epoch = 1

    # --------------------------------------
    # Resume training if requested
    # --------------------------------------

    checkpoint_path = (
        CHECKPOINT_DIR / "latest_checkpoint.pth"
    )

    if RESUME and checkpoint_path.exists():

        print("\nLoading previous checkpoint...")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = checkpoint["epoch"] + 1

        best_dice = checkpoint["best_dice"]

        print("Resuming from epoch:", start_epoch)

    # --------------------------------------
    # Training loop
    # --------------------------------------

    for epoch in range(start_epoch, NUM_EPOCHS + 1):

        print("\n" + "=" * 50)
        print(f"STARTING EPOCH {epoch}/{NUM_EPOCHS}")
        print("=" * 50)

        train_loss, train_time = train_one_epoch(
            model,
            train_loader,
            optimizer,
            epoch
        )

        # ----------------------------------
        # Validation
        # ----------------------------------

        print("\nRunning validation...")

        val_start = time.perf_counter()

        val_loss, val_dice = validate(
            model,
            val_loader
        )

        val_time = time.perf_counter() - val_start

        # ----------------------------------
        # Display results
        # ----------------------------------

        print("\nEPOCH RESULTS")
        print("-" * 40)

        print(f"Training loss:   {train_loss:.6f}")
        print(f"Validation loss: {val_loss:.6f}")
        print(f"Validation Dice: {val_dice:.6f}")

        print(f"Training time:   {train_time / 3600:.2f} hours")
        print(f"Validation time: {val_time / 60:.2f} minutes")

        # ----------------------------------
        # Save best model
        # ----------------------------------

        if val_dice > best_dice:

            best_dice = val_dice

            torch.save(
                model.state_dict(),
                CHECKPOINT_DIR / "best_model.pth"
            )

            print("New best model saved!")

        # ----------------------------------
        # Save latest checkpoint
        # ----------------------------------

        save_checkpoint(
            model,
            optimizer,
            epoch,
            val_dice,
            best_dice
        )

        print("\nLatest checkpoint saved.")

        print(f"Best validation Dice so far: {best_dice:.6f}")

    print("\nTRAINING COMPLETED!")
    print("================================")


if __name__ == "__main__":
    main()

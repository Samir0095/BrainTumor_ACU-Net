
import sys
from pathlib import Path

# ==========================================
# Project Path
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import time
import torch
import torch.nn as nn
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

BATCH_SIZE = 1
LEARNING_RATE = 0.0001
NUM_BATCHES = 100

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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

    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets
    )

    dice = dice_loss(logits, targets)

    return bce + dice, bce.item(), dice.item()


# ==========================================
# 4. Main Training Function
# ==========================================

def main():

    print("Starting ACU-Net 100-batch training test")
    print("----------------------------------------")

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":

        print("GPU:", torch.cuda.get_device_name(0))

        torch.cuda.reset_peak_memory_stats()

    # --------------------------------------
    # Load datasets
    # --------------------------------------

    print("\nLoading datasets...")

    train_loader, val_loader = get_dataloaders(
        DATASET_PATH,
        batch_size=BATCH_SIZE
    )

    # --------------------------------------
    # Initialize model
    # --------------------------------------

    print("\nInitializing ACU-Net...")

    model = ACUNet().to(DEVICE)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Total parameters: {total_parameters:,}")

    # --------------------------------------
    # Optimizer
    # --------------------------------------

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------
    # Training setup
    # --------------------------------------

    model.train()

    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0

    completed_batches = 0

    print(f"\nStarting training for {NUM_BATCHES} batches...")

    start_time = time.perf_counter()

    # --------------------------------------
    # Training loop
    # --------------------------------------

    for batch_idx, (images, masks) in enumerate(train_loader):

        if batch_idx >= NUM_BATCHES:
            break

        images = images.to(DEVICE)

        masks = masks.unsqueeze(1).to(DEVICE)

        # Forward and backward passes

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)

        loss, bce, dice = combined_loss(
            logits,
            masks
        )

        loss.backward()

        optimizer.step()

        # Accumulate losses

        total_loss += loss.item()
        total_bce += bce
        total_dice += dice

        completed_batches += 1

        # Print progress every 10 batches

        if completed_batches % 10 == 0:

            print(
                f"Batch [{completed_batches}/{NUM_BATCHES}] "
                f"Loss: {loss.item():.6f}"
            )

    # --------------------------------------
    # Calculate training time
    # --------------------------------------

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    elapsed_time = time.perf_counter() - start_time

    # --------------------------------------
    # Calculate average losses
    # --------------------------------------

    average_loss = total_loss / completed_batches
    average_bce = total_bce / completed_batches
    average_dice = total_dice / completed_batches

    average_batch_time = elapsed_time / completed_batches

    # --------------------------------------
    # Display results
    # --------------------------------------

    print("\n100-BATCH TRAINING TEST COMPLETED!")
    print("----------------------------------")

    print(f"Completed batches: {completed_batches}")

    print(f"Average total loss: {average_loss:.6f}")
    print(f"Average BCE loss:   {average_bce:.6f}")
    print(f"Average Dice loss:  {average_dice:.6f}")

    print(f"\nTotal training time: {elapsed_time:.2f} seconds")

    print(
        f"Average time per batch: "
        f"{average_batch_time:.2f} seconds"
    )

    # --------------------------------------
    # GPU memory usage
    # --------------------------------------

    if DEVICE.type == "cuda":

        peak_memory = (
            torch.cuda.max_memory_allocated() / 1024**3
        )

        reserved_memory = (
            torch.cuda.max_memory_reserved() / 1024**3
        )

        print(f"\nPeak GPU memory allocated: {peak_memory:.2f} GB")

        print(f"Peak GPU memory reserved:  {reserved_memory:.2f} GB")

    print("\n100-BATCH TRAINING TEST PASSED!")


# ==========================================
# 5. Run Main
# ==========================================

if __name__ == "__main__":
    main()

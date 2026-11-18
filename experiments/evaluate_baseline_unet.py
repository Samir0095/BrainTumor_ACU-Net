import sys
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from models.unet_baseline import UNet
from data.dataloader import get_dataloaders

# Configuration
DATASET_PATH = (
    r"E:\3.2\Brain Tumor Research\BrainTumor_UNet"
    r"\Datasets\BraTS2018"
)

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "baseline_best_model.pth"

OUTPUT_DIR = PROJECT_ROOT / "results" / "baseline_unet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 1
THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    print("Loading Baseline U-Net...")

    model = UNet().to(DEVICE)

    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("Baseline checkpoint loaded successfully.")

    return model


def evaluate():

    print("\nBaseline U-Net Full Validation Evaluation")
    print("=" * 45)

    print("Device:", DEVICE)
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Threshold:", THRESHOLD)

    model = load_model()

    print("\nLoading validation dataset...")

    _, val_loader = get_dataloaders(
        DATASET_PATH,
        batch_size=BATCH_SIZE
    )

    print("Validation batches:", len(val_loader))

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    total_slices = 0
    tumor_slices = 0
    empty_slices = 0

    empty_slices_with_false_positives = 0

    tumor_slice_dice_scores = []
    all_slice_dice_scores = []

    slice_results = []

    print("\nStarting evaluation...")

    with torch.no_grad():

        for batch_idx, (images, masks) in enumerate(val_loader):

            images = images.to(DEVICE)
            masks = masks.unsqueeze(1).to(DEVICE)

            logits = model(images)

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities > THRESHOLD
            ).float()

            tp = (
                (predictions == 1) &
                (masks == 1)
            ).sum().item()

            fp = (
                (predictions == 1) &
                (masks == 0)
            ).sum().item()

            fn = (
                (predictions == 0) &
                (masks == 1)
            ).sum().item()

            tn = (
                (predictions == 0) &
                (masks == 0)
            ).sum().item()

            TP += tp
            FP += fp
            FN += fn
            TN += tn

            for sample_idx in range(images.size(0)):

                sample_prediction = predictions[sample_idx]
                sample_mask = masks[sample_idx]

                gt_pixels = sample_mask.sum().item()
                predicted_pixels = sample_prediction.sum().item()

                intersection = (
                    sample_prediction * sample_mask
                ).sum().item()

                denominator = predicted_pixels + gt_pixels

                if denominator == 0:
                    dice = 1.0
                else:
                    dice = (2.0 * intersection) / denominator

                total_slices += 1

                if gt_pixels > 0:

                    tumor_slices += 1
                    tumor_slice_dice_scores.append(dice)

                else:

                    empty_slices += 1

                    if predicted_pixels > 0:
                        empty_slices_with_false_positives += 1

                all_slice_dice_scores.append(dice)

                slice_results.append({
                    "slice": total_slices,
                    "ground_truth_pixels": int(gt_pixels),
                    "predicted_pixels": int(predicted_pixels),
                    "dice": dice
                })

            if (batch_idx + 1) % 1000 == 0:

                print(
                    f"Processed {batch_idx + 1} "
                    f"/ {len(val_loader)} batches"
                )

    # Calculate metrics

    global_dice = (
        (2 * TP) / (2 * TP + FP + FN)
        if (2 * TP + FP + FN) > 0
        else 0.0
    )

    precision = (
        TP / (TP + FP)
        if (TP + FP) > 0
        else 0.0
    )

    recall = (
        TP / (TP + FN)
        if (TP + FN) > 0
        else 0.0
    )

    specificity = (
        TN / (TN + FP)
        if (TN + FP) > 0
        else 0.0
    )

    mean_tumor_dice = (
        sum(tumor_slice_dice_scores) /
        len(tumor_slice_dice_scores)
        if tumor_slice_dice_scores
        else 0.0
    )

    mean_all_slice_dice = (
        sum(all_slice_dice_scores) /
        len(all_slice_dice_scores)
        if all_slice_dice_scores
        else 0.0
    )

    empty_slice_false_positive_rate = (
        empty_slices_with_false_positives / empty_slices
        if empty_slices > 0
        else 0.0
    )

    # Print results

    print("\n")
    print("=" * 55)
    print("BASELINE U-NET VALIDATION RESULTS")
    print("=" * 55)

    print("\nDataset Summary")

    print("Total validation slices:", total_slices)
    print("Tumor-containing slices:", tumor_slices)
    print("Empty slices:", empty_slices)

    print("\nPixel-level Confusion Counts")

    print("TP:", TP)
    print("FP:", FP)
    print("FN:", FN)
    print("TN:", TN)

    print("\nSegmentation Metrics")

    print(f"Global Dice: {global_dice:.6f}")
    print(f"Mean Tumor-slice Dice: {mean_tumor_dice:.6f}")
    print(f"Mean All-slice Dice: {mean_all_slice_dice:.6f}")

    print(f"Precision: {precision:.6f}")
    print(f"Recall / Sensitivity: {recall:.6f}")
    print(f"Specificity: {specificity:.6f}")

    print("\nEmpty Slice Analysis")

    print(
        "Empty slices with false-positive predictions:",
        empty_slices_with_false_positives
    )

    print(
        f"Empty-slice false-positive rate: "
        f"{empty_slice_false_positive_rate:.6f}"
    )

    print("=" * 55)

    # Save summary

    summary_path = OUTPUT_DIR / "evaluation_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as file:

        file.write("Baseline U-Net Full Validation Evaluation\n")
        file.write("=" * 45 + "\n\n")

        file.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
        file.write(f"Threshold: {THRESHOLD}\n\n")

        file.write(f"Total slices: {total_slices}\n")
        file.write(f"Tumor slices: {tumor_slices}\n")
        file.write(f"Empty slices: {empty_slices}\n\n")

        file.write(f"TP: {TP}\n")
        file.write(f"FP: {FP}\n")
        file.write(f"FN: {FN}\n")
        file.write(f"TN: {TN}\n\n")

        file.write(f"Global Dice: {global_dice:.6f}\n")
        file.write(f"Mean tumor-slice Dice: {mean_tumor_dice:.6f}\n")
        file.write(f"Mean all-slice Dice: {mean_all_slice_dice:.6f}\n")

        file.write(f"Precision: {precision:.6f}\n")
        file.write(f"Recall: {recall:.6f}\n")
        file.write(f"Specificity: {specificity:.6f}\n")

        file.write(
            f"Empty-slice false-positive rate: "
            f"{empty_slice_false_positive_rate:.6f}\n"
        )

    # Save per-slice CSV

    csv_path = OUTPUT_DIR / "slice_evaluation.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "slice",
                "ground_truth_pixels",
                "predicted_pixels",
                "dice"
            ]
        )

        writer.writeheader()
        writer.writerows(slice_results)

    print("\nEvaluation files saved successfully!")

    print("Summary:", summary_path)
    print("Per-slice results:", csv_path)


if __name__ == "__main__":
    evaluate()
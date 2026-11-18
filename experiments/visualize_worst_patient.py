import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt

# ============================================================
# Project configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.unet_baseline import UNet
from models.acu_net import ACUNet

DATASET_ROOT = Path(
    r"E:\3.2\Brain Tumor Research\BrainTumor_UNet"
    r"\Datasets\BraTS2018"
)

PATIENT_ID = "Brats18_TCIA10_202_1"

PATIENT_DIR = (
    DATASET_ROOT
    / "MICCAI_BraTS_2018_Data_Training"
    / "LGG"
    / PATIENT_ID
)

BASELINE_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "baseline_best_model.pth"
)

ACU_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "best_model.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "results"
    / "patient_error_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.5
TOP_K = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Load MRI data
# ============================================================

def normalize_volume(volume):
    """
    Normalize each MRI modality using nonzero voxels.
    Matches the baseline dataloader normalization.
    """

    volume = volume.astype(np.float32)

    nonzero_mask = volume != 0

    if nonzero_mask.sum() == 0:
        return volume

    mean = volume[nonzero_mask].mean()
    std = volume[nonzero_mask].std()

    if std > 0:
        volume[nonzero_mask] = (
            volume[nonzero_mask] - mean
        ) / std

    volume[~nonzero_mask] = 0

    return volume


def load_patient():

    modalities = ["flair", "t1", "t1ce", "t2"]

    image_volumes = []

    for modality in modalities:

        path = PATIENT_DIR / f"{PATIENT_ID}_{modality}.nii"

        volume = nib.load(str(path)).get_fdata()

        volume = normalize_volume(volume)

        image_volumes.append(volume)

    image = np.stack(image_volumes, axis=0)

    seg_path = PATIENT_DIR / f"{PATIENT_ID}_seg.nii"

    segmentation = nib.load(
        str(seg_path)
    ).get_fdata()

    # Convert BraTS tumor labels to binary.
    segmentation = (segmentation > 0).astype(np.float32)

    print("Image volume shape:", image.shape)
    print("Segmentation shape:", segmentation.shape)

    return image, segmentation


# ============================================================
# Load models
# ============================================================

def load_models():

    print("Loading Baseline U-Net...")

    baseline_model = UNet().to(DEVICE)

    baseline_state = torch.load(
        BASELINE_CHECKPOINT,
        map_location=DEVICE
    )

    baseline_model.load_state_dict(
        baseline_state,
        strict=True
    )

    baseline_model.eval()

    print("Baseline checkpoint loaded.")

    print("\nLoading ACU-Net...")

    acu_model = ACUNet().to(DEVICE)

    acu_state = torch.load(
        ACU_CHECKPOINT,
        map_location=DEVICE
    )

    acu_model.load_state_dict(
        acu_state
    )

    acu_model.eval()

    print("ACU-Net checkpoint loaded.")

    return baseline_model, acu_model


# ============================================================
# Dice calculation
# ============================================================

def calculate_dice(prediction, ground_truth):

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    denominator = (
        prediction.sum() + ground_truth.sum()
    )

    if denominator == 0:
        return 1.0

    return (
        2.0 * intersection / denominator
    )


# ============================================================
# Evaluate patient slices
# ============================================================

def evaluate_patient(
    image,
    segmentation,
    baseline_model,
    acu_model
):

    slice_results = []

    number_of_slices = image.shape[-1]

    print("\nEvaluating patient slices...")
    print("Total slices:", number_of_slices)

    with torch.no_grad():

        for slice_idx in range(number_of_slices):

            # Input shape: [4, 240, 240]
            image_slice = image[:, :, :, slice_idx]

            # Convert to tensor: [1, 4, 240, 240]
            input_tensor = torch.from_numpy(
                image_slice
            ).unsqueeze(0).float().to(DEVICE)

            # Ground truth: [240, 240]
            ground_truth = segmentation[:, :, slice_idx]

            # ------------------------------------------------
            # Baseline prediction
            # ------------------------------------------------

            baseline_logits = baseline_model(
                input_tensor
            )

            baseline_probabilities = torch.sigmoid(
                baseline_logits
            )

            baseline_prediction = (
                baseline_probabilities > THRESHOLD
            ).squeeze().cpu().numpy()

            # ------------------------------------------------
            # ACU-Net prediction
            # ------------------------------------------------

            acu_logits = acu_model(
                input_tensor
            )

            acu_probabilities = torch.sigmoid(
                acu_logits
            )

            acu_prediction = (
                acu_probabilities > THRESHOLD
            ).squeeze().cpu().numpy()

            # ------------------------------------------------
            # Calculate Dice
            # ------------------------------------------------

            baseline_dice = calculate_dice(
                baseline_prediction,
                ground_truth.astype(bool)
            )

            acu_dice = calculate_dice(
                acu_prediction,
                ground_truth.astype(bool)
            )

            delta = acu_dice - baseline_dice

            slice_results.append({
                "slice_idx": slice_idx,
                "ground_truth": ground_truth,
                "flair": image_slice[0],
                "baseline_prediction": baseline_prediction,
                "acu_prediction": acu_prediction,
                "baseline_dice": baseline_dice,
                "acu_dice": acu_dice,
                "delta": delta,
            })

            if (slice_idx + 1) % 25 == 0:

                print(
                    f"Processed {slice_idx + 1}"
                    f" / {number_of_slices} slices"
                )

    return slice_results


# ============================================================
# Select worst slices
# ============================================================

def select_worst_slices(slice_results):

    tumor_slices = [
        result
        for result in slice_results
        if result["ground_truth"].sum() > 0
    ]

    # ACU-Net's most negative Dice differences.
    worsened_slices = [
        result
        for result in tumor_slices
        if result["delta"] < 0
    ]

    worsened_slices.sort(
        key=lambda result: result["delta"]
    )

    selected = worsened_slices[:TOP_K]

    print("\nWorst slices:")

    for result in selected:

        print(
            f"Slice {result['slice_idx']} | "
            f"Baseline Dice: {result['baseline_dice']:.4f} | "
            f"ACU Dice: {result['acu_dice']:.4f} | "
            f"Delta: {result['delta']:.4f}"
        )

    return selected


# ============================================================
# Visualize results
# ============================================================

def visualize_slices(selected_slices):

    if not selected_slices:

        print(
            "No worsened tumor-containing slices found."
        )

        return

    rows = len(selected_slices)

    fig, axes = plt.subplots(
        rows,
        4,
        figsize=(16, 4 * rows)
    )

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, result in enumerate(selected_slices):

        slice_idx = result["slice_idx"]

        flair = result["flair"]
        ground_truth = result["ground_truth"]
        baseline_prediction = result["baseline_prediction"]
        acu_prediction = result["acu_prediction"]

        # FLAIR
        axes[row, 0].imshow(
            flair,
            cmap="gray"
        )

        axes[row, 0].set_title(
            f"FLAIR | Slice {slice_idx}"
        )

        # Ground truth
        axes[row, 1].imshow(
            flair,
            cmap="gray"
        )

        axes[row, 1].imshow(
            ground_truth,
            cmap="Reds",
            alpha=0.5
        )

        axes[row, 1].set_title(
            "Ground Truth"
        )

        # Baseline prediction
        axes[row, 2].imshow(
            flair,
            cmap="gray"
        )

        axes[row, 2].imshow(
            baseline_prediction,
            cmap="Blues",
            alpha=0.5
        )

        axes[row, 2].set_title(
            f"Baseline U-Net\n"
            f"Dice: {result['baseline_dice']:.4f}"
        )

        # ACU-Net prediction
        axes[row, 3].imshow(
            flair,
            cmap="gray"
        )

        axes[row, 3].imshow(
            acu_prediction,
            cmap="Greens",
            alpha=0.5
        )

        axes[row, 3].set_title(
            f"ACU-Net\n"
            f"Dice: {result['acu_dice']:.4f}\n"
            f"Delta: {result['delta']:.4f}"
        )

        for col in range(4):

            axes[row, col].axis("off")

    plt.suptitle(
        f"Worst Patient Error Analysis: {PATIENT_ID}",
        fontsize=16
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{PATIENT_ID}_worst_slices.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("\nVisualization saved to:")
    print(output_path)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("WORST PATIENT VISUAL ERROR ANALYSIS")
    print("=" * 60)

    print("Patient:", PATIENT_ID)
    print("Device:", DEVICE)

    image, segmentation = load_patient()

    baseline_model, acu_model = load_models()

    slice_results = evaluate_patient(
        image,
        segmentation,
        baseline_model,
        acu_model
    )

    selected_slices = select_worst_slices(
        slice_results
    )

    visualize_slices(selected_slices)


if __name__ == "__main__":
    main()
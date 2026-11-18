import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataloader import get_patient_split

DATASET_PATH = (
    r"E:\3.2\Brain Tumor Research\BrainTumor_UNet"
    r"\Datasets\BraTS2018"
)

BASELINE_CSV = (
    PROJECT_ROOT / "results" / "baseline_unet" / "slice_evaluation.csv"
)

ACU_CSV = PROJECT_ROOT / "results" / "slice_evaluation.csv"

OUTPUT_CSV = (
    PROJECT_ROOT / "results" / "patient_wise_error_analysis.csv"
)


# --------------------------------------------------
# Load evaluation results
# --------------------------------------------------

baseline = pd.read_csv(BASELINE_CSV)
acu = pd.read_csv(ACU_CSV)

required_columns = {
    "slice",
    "ground_truth_pixels",
    "predicted_pixels",
    "dice",
}

for name, df in [("Baseline", baseline), ("ACU-Net", acu)]:
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"{name} CSV is missing columns: {missing}")

assert len(baseline) == len(acu), "CSV row counts differ!"

assert (
    baseline["ground_truth_pixels"].to_numpy()
    == acu["ground_truth_pixels"].to_numpy()
).all(), "Ground truth pixel counts differ!"

# --------------------------------------------------
# Reconstruct validation patient order
# --------------------------------------------------

_, val_patients = get_patient_split(DATASET_PATH)

patient_ids = []
slice_counts = []

for patient_path in val_patients:

    patient_id = patient_path.name

    flair_path = patient_path / f"{patient_id}_flair.nii"

    flair_img = nib.load(flair_path)

    num_slices = flair_img.shape[2]

    patient_ids.append(patient_id)
    slice_counts.append(num_slices)

assert sum(slice_counts) == len(baseline), (
    "Reconstructed slice count does not match CSV!"
)

# --------------------------------------------------
# Map slices to patients
# --------------------------------------------------

slice_patient_ids = []

for patient_id, num_slices in zip(patient_ids, slice_counts):
    slice_patient_ids.extend([patient_id] * num_slices)

baseline["patient_id"] = slice_patient_ids
acu["patient_id"] = slice_patient_ids

# --------------------------------------------------
# Calculate patient-wise pixel statistics
# --------------------------------------------------

baseline_area = (
    baseline.groupby("patient_id")
    .agg(
        GT_Pixels=("ground_truth_pixels", "sum"),
        Baseline_Predicted_Pixels=("predicted_pixels", "sum"),
    )
)

acu_area = (
    acu.groupby("patient_id")
    .agg(
        ACU_Predicted_Pixels=("predicted_pixels", "sum"),
    )
)

patient_results = baseline_area.join(acu_area)

# --------------------------------------------------
# Calculate mean tumor-slice Dice per patient
# --------------------------------------------------

baseline_tumor = baseline[
    baseline["ground_truth_pixels"] > 0
]

acu_tumor = acu[
    acu["ground_truth_pixels"] > 0
]

baseline_dice = (
    baseline_tumor.groupby("patient_id")["dice"].mean()
)

acu_dice = (
    acu_tumor.groupby("patient_id")["dice"].mean()
)

patient_results["Baseline_Dice"] = baseline_dice
patient_results["ACU_Net_Dice"] = acu_dice

patient_results["Dice_Improvement"] = (
    patient_results["ACU_Net_Dice"]
    - patient_results["Baseline_Dice"]
)

# --------------------------------------------------
# Calculate prediction-to-ground-truth area ratios
# --------------------------------------------------

patient_results["Baseline_Area_Ratio"] = (
    patient_results["Baseline_Predicted_Pixels"]
    / patient_results["GT_Pixels"]
)

patient_results["ACU_Area_Ratio"] = (
    patient_results["ACU_Predicted_Pixels"]
    / patient_results["GT_Pixels"]
)

# --------------------------------------------------
# Classify patient-wise performance
# --------------------------------------------------

def classify_patient(delta):

    if np.isclose(delta, 0.0, atol=1e-12):
        return "Tied"

    elif delta > 0:
        return "Improved"

    return "Worsened"


patient_results["Performance"] = (
    patient_results["Dice_Improvement"]
    .apply(classify_patient)
)

# --------------------------------------------------
# Save results
# --------------------------------------------------

patient_results.index.name = "Patient_ID"

patient_results.to_csv(OUTPUT_CSV)

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n===== PATIENT-WISE ERROR ANALYSIS =====")

print("Validation patients:", len(patient_results))

print(
    "Patients improved:",
    (patient_results["Performance"] == "Improved").sum()
)

print(
    "Patients worsened:",
    (patient_results["Performance"] == "Worsened").sum()
)

print(
    "Patients tied:",
    (patient_results["Performance"] == "Tied").sum()
)

print("\n===== AGGREGATE TUMOR PIXELS =====")

print(
    "Ground-truth pixels:",
    int(patient_results["GT_Pixels"].sum())
)

print(
    "Baseline predicted pixels:",
    int(patient_results["Baseline_Predicted_Pixels"].sum())
)

print(
    "ACU-Net predicted pixels:",
    int(patient_results["ACU_Predicted_Pixels"].sum())
)

print("\n===== MEAN AREA RATIOS =====")

print(
    "Baseline:",
    patient_results["Baseline_Area_Ratio"].mean()
)

print(
    "ACU-Net:",
    patient_results["ACU_Area_Ratio"].mean()
)

print("\n===== TOP 5 WORST ACU-NET PATIENTS =====")

print(
    patient_results.sort_values("Dice_Improvement")
    .head(5)[
        [
            "GT_Pixels",
            "Baseline_Predicted_Pixels",
            "ACU_Predicted_Pixels",
            "Baseline_Dice",
            "ACU_Net_Dice",
            "Dice_Improvement",
            "Baseline_Area_Ratio",
            "ACU_Area_Ratio",
        ]
    ].to_string()
)

print("\nResults saved to:")
print(OUTPUT_CSV)

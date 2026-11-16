import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataloader import get_patient_split

DATASET_PATH = (
    r"E:\3.2\Brain Tumor Research\BrainTumor_UNet"
    r"\Datasets\BraTS2018"
)

BASELINE_CSV = PROJECT_ROOT / "results" / "baseline_unet" / "slice_evaluation.csv"
ACU_CSV = PROJECT_ROOT / "results" / "slice_evaluation.csv"

# --------------------------------------------------
# Load evaluation results
# --------------------------------------------------

baseline = pd.read_csv(BASELINE_CSV)
acu = pd.read_csv(ACU_CSV)

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
    "Reconstructed slice count does not match evaluation CSV!"
)

# --------------------------------------------------
# Map each slice to its patient
# --------------------------------------------------

slice_patient_ids = []

for patient_id, num_slices in zip(patient_ids, slice_counts):
    slice_patient_ids.extend([patient_id] * num_slices)

baseline["patient_id"] = slice_patient_ids
acu["patient_id"] = slice_patient_ids

# --------------------------------------------------
# Keep tumor-containing slices only
# --------------------------------------------------

tumor_mask = baseline["ground_truth_pixels"] > 0

baseline_tumor = baseline[tumor_mask].copy()
acu_tumor = acu[tumor_mask].copy()

# --------------------------------------------------
# Calculate mean tumor-slice Dice per patient
# --------------------------------------------------

baseline_patient = (
    baseline_tumor.groupby("patient_id")["dice"].mean()
)

acu_patient = (
    acu_tumor.groupby("patient_id")["dice"].mean()
)

patient_results = pd.DataFrame({
    "Baseline_Dice": baseline_patient,
    "ACU_Net_Dice": acu_patient,
})

patient_results["Dice_Improvement"] = (
    patient_results["ACU_Net_Dice"]
    - patient_results["Baseline_Dice"]
)

differences = patient_results["Dice_Improvement"].to_numpy()

# --------------------------------------------------
# Patient-level statistics
# --------------------------------------------------

print("\n===== PATIENT-LEVEL ANALYSIS =====")

print("Validation patients:", len(val_patients))
print("Patients with tumor-containing slices:", len(patient_results))

print(
    f"Baseline mean patient Dice: "
    f"{patient_results['Baseline_Dice'].mean():.6f}"
)

print(
    f"ACU-Net mean patient Dice: "
    f"{patient_results['ACU_Net_Dice'].mean():.6f}"
)

print(
    f"Mean patient-level improvement: "
    f"{differences.mean():+.6f}"
)

print(
    f"Median patient-level improvement: "
    f"{np.median(differences):+.6f}"
)

print("Improved patients:", (differences > 0).sum())
print("Worsened patients:", (differences < 0).sum())
print("Tied patients:", (differences == 0).sum())

# --------------------------------------------------
# Patient-level Wilcoxon signed-rank test
# --------------------------------------------------

statistic, p_value = wilcoxon(
    differences,
    alternative="greater"
)

print("\n===== PATIENT-LEVEL WILCOXON TEST =====")
print("W statistic:", statistic)
print("P-value:", p_value)

# --------------------------------------------------
# Bootstrap 95% confidence interval
# --------------------------------------------------

rng = np.random.default_rng(42)

n_patients = len(differences)
n_bootstrap = 10000

bootstrap_means = np.empty(n_bootstrap)

for i in range(n_bootstrap):
    sampled_differences = rng.choice(
        differences,
        size=n_patients,
        replace=True
    )

    bootstrap_means[i] = sampled_differences.mean()

ci_lower, ci_upper = np.percentile(
    bootstrap_means,
    [2.5, 97.5]
)

print("\n===== PATIENT-LEVEL BOOTSTRAP =====")
print("95% Confidence Interval:")
print(f"Lower bound: {ci_lower:.6f}")
print(f"Upper bound: {ci_upper:.6f}")

# --------------------------------------------------
# Save results
# --------------------------------------------------

output_path = PROJECT_ROOT / "results" / "patient_level_comparison.csv"

patient_results.to_csv(output_path)

print("\nPatient-level results saved to:")
print(output_path)
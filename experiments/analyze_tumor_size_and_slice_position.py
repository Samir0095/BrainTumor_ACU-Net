
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ============================================================
# Project configuration
# ============================================================

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

OUTPUT_DIR = PROJECT_ROOT / "results" / "tumor_size_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Reconstruct TP, FP and FN from saved Dice
# ============================================================

def reconstruct_components(df, model_name):

    gt = df["ground_truth_pixels"].to_numpy(dtype=np.float64)
    pred = df["predicted_pixels"].to_numpy(dtype=np.float64)
    dice = df["dice"].to_numpy(dtype=np.float64)

    denominator = gt + pred

    tp_raw = np.zeros(len(df), dtype=np.float64)

    valid = denominator > 0

    tp_raw[valid] = (
        dice[valid] * denominator[valid] / 2.0
    )

    tp = np.rint(tp_raw)

    if not np.allclose(tp_raw, tp, atol=1e-4):
        raise ValueError(
            f"{model_name}: TP reconstruction failed."
        )

    tp = tp.astype(np.int64)

    if np.any(tp < 0) or np.any(tp > gt) or np.any(tp > pred):
        raise ValueError(
            f"{model_name}: invalid TP counts found."
        )

    fn = gt.astype(np.int64) - tp
    fp = pred.astype(np.int64) - tp

    reconstructed_dice = np.ones(len(df), dtype=np.float64)

    reconstructed_dice[valid] = (
        2.0 * tp[valid] / denominator[valid]
    )

    if not np.allclose(reconstructed_dice, dice, atol=1e-5):
        raise ValueError(
            f"{model_name}: Dice consistency check failed."
        )

    return pd.DataFrame({
        "TP": tp,
        "FP": fp,
        "FN": fn,
    })


# ============================================================
# Load evaluation CSVs
# ============================================================

baseline = pd.read_csv(BASELINE_CSV)
acu = pd.read_csv(ACU_CSV)

assert len(baseline) == len(acu), (
    "Baseline and ACU-Net CSV row counts differ."
)

assert np.array_equal(
    baseline["ground_truth_pixels"].to_numpy(),
    acu["ground_truth_pixels"].to_numpy()
), "Ground-truth pixel counts differ."

# ============================================================
# Reconstruct validation patient order
# ============================================================

_, val_patients = get_patient_split(DATASET_PATH)

patient_ids = []
slice_counts = []

for patient_path in val_patients:

    patient_id = patient_path.name

    flair_path = (
        patient_path / f"{patient_id}_flair.nii"
    )

    flair_img = nib.load(str(flair_path))

    patient_ids.append(patient_id)
    slice_counts.append(flair_img.shape[2])

assert sum(slice_counts) == len(baseline), (
    "Reconstructed patient slice count does not match CSV."
)

slice_patient_ids = []

for patient_id, count in zip(patient_ids, slice_counts):
    slice_patient_ids.extend([patient_id] * count)

baseline["patient_id"] = slice_patient_ids
acu["patient_id"] = slice_patient_ids

# ============================================================
# Reconstruct confusion components
# ============================================================

baseline_components = reconstruct_components(
    baseline,
    "Baseline U-Net"
)

acu_components = reconstruct_components(
    acu,
    "ACU-Net"
)

baseline = pd.concat(
    [baseline, baseline_components],
    axis=1
)

acu = pd.concat(
    [acu, acu_components],
    axis=1
)

# ============================================================
# Build slice-level analysis table
# ============================================================

slices = pd.DataFrame({
    "patient_id": slice_patient_ids,
    "slice_index": np.tile(
        np.concatenate([
            np.arange(count) for count in slice_counts
        ]),
        1
    ),
    "GT_Pixels": baseline["ground_truth_pixels"],
    "Baseline_Dice": baseline["dice"],
    "ACU_Dice": acu["dice"],
    "Baseline_TP": baseline["TP"],
    "Baseline_FP": baseline["FP"],
    "Baseline_FN": baseline["FN"],
    "ACU_TP": acu["TP"],
    "ACU_FP": acu["FP"],
    "ACU_FN": acu["FN"],
})

# Keep tumor-containing slices for tumor-size and position analysis.
tumor_slices = slices[
    slices["GT_Pixels"] > 0
].copy()

tumor_slices["Dice_Change"] = (
    tumor_slices["ACU_Dice"]
    - tumor_slices["Baseline_Dice"]
)

# ============================================================
# Categorize tumor size using quartiles
# ============================================================

q25 = tumor_slices["GT_Pixels"].quantile(0.25)
q75 = tumor_slices["GT_Pixels"].quantile(0.75)

def classify_tumor_size(pixel_count):

    if pixel_count <= q25:
        return "Small"

    elif pixel_count <= q75:
        return "Medium"

    return "Large"


tumor_slices["Tumor_Size_Group"] = (
    tumor_slices["GT_Pixels"].apply(classify_tumor_size)
)

# ============================================================
# Analyze tumor size groups
# ============================================================

size_rows = []

for group in ["Small", "Medium", "Large"]:

    subset = tumor_slices[
        tumor_slices["Tumor_Size_Group"] == group
    ]

    if subset.empty:
        continue

    size_rows.append({
        "Tumor_Size_Group": group,
        "Slice_Count": len(subset),
        "Mean_GT_Pixels": subset["GT_Pixels"].mean(),
        "Baseline_Mean_Dice": subset["Baseline_Dice"].mean(),
        "ACU_Mean_Dice": subset["ACU_Dice"].mean(),
        "Mean_Dice_Change": subset["Dice_Change"].mean(),
        "Baseline_TP": subset["Baseline_TP"].sum(),
        "Baseline_FP": subset["Baseline_FP"].sum(),
        "Baseline_FN": subset["Baseline_FN"].sum(),
        "ACU_TP": subset["ACU_TP"].sum(),
        "ACU_FP": subset["ACU_FP"].sum(),
        "ACU_FN": subset["ACU_FN"].sum(),
    })


size_results = pd.DataFrame(size_rows)

size_results.to_csv(
    OUTPUT_DIR / "tumor_size_group_analysis.csv",
    index=False
)

# ============================================================
# Categorize position within each patient's tumor sequence
# ============================================================

tumor_slices = tumor_slices.sort_values(
    ["patient_id", "slice_index"]
).copy()

tumor_slices["Tumor_Slice_Order"] = (
    tumor_slices.groupby("patient_id").cumcount() + 1
)

tumor_slices["Tumor_Slice_Count"] = (
    tumor_slices.groupby("patient_id")["slice_index"]
    .transform("count")
)

tumor_slices["Relative_Position"] = (
    tumor_slices["Tumor_Slice_Order"]
    / tumor_slices["Tumor_Slice_Count"]
)

def classify_position(position):

    if position <= 1 / 3:
        return "Beginning"

    elif position <= 2 / 3:
        return "Middle"

    return "End"


tumor_slices["Position_Group"] = (
    tumor_slices["Relative_Position"]
    .apply(classify_position)
)

# ============================================================
# Analyze slice position groups
# ============================================================

position_rows = []

for group in ["Beginning", "Middle", "End"]:

    subset = tumor_slices[
        tumor_slices["Position_Group"] == group
    ]

    if subset.empty:
        continue

    position_rows.append({
        "Position_Group": group,
        "Slice_Count": len(subset),
        "Mean_GT_Pixels": subset["GT_Pixels"].mean(),
        "Baseline_Mean_Dice": subset["Baseline_Dice"].mean(),
        "ACU_Mean_Dice": subset["ACU_Dice"].mean(),
        "Mean_Dice_Change": subset["Dice_Change"].mean(),
        "Baseline_FN": subset["Baseline_FN"].sum(),
        "ACU_FN": subset["ACU_FN"].sum(),
        "Baseline_FP": subset["Baseline_FP"].sum(),
        "ACU_FP": subset["ACU_FP"].sum(),
    })


position_results = pd.DataFrame(position_rows)

position_results.to_csv(
    OUTPUT_DIR / "slice_position_analysis.csv",
    index=False
)

# ============================================================
# Patient-level tumor burden analysis
# ============================================================

patient_results = (
    tumor_slices.groupby("patient_id")
    .agg(
        Total_GT_Pixels=("GT_Pixels", "sum"),
        Mean_Tumor_Slice_Pixels=("GT_Pixels", "mean"),
        Baseline_Mean_Dice=("Baseline_Dice", "mean"),
        ACU_Mean_Dice=("ACU_Dice", "mean"),
    )
)

patient_results["Dice_Improvement"] = (
    patient_results["ACU_Mean_Dice"]
    - patient_results["Baseline_Mean_Dice"]
)

patient_results.to_csv(
    OUTPUT_DIR / "patient_tumor_burden_analysis.csv"
)

# ============================================================
# Spearman correlation
# ============================================================

total_burden_corr = spearmanr(
    patient_results["Total_GT_Pixels"],
    patient_results["Dice_Improvement"]
)

mean_slice_burden_corr = spearmanr(
    patient_results["Mean_Tumor_Slice_Pixels"],
    patient_results["Dice_Improvement"]
)

# ============================================================
# Save complete slice-level dataset
# ============================================================

tumor_slices.to_csv(
    OUTPUT_DIR / "tumor_slice_analysis.csv",
    index=False
)

# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 65)
print("STEP 12.5: TUMOR SIZE AND SLICE-WISE FAILURE ANALYSIS")
print("=" * 65)

print("\nValidation patients:", len(patient_results))
print("Tumor-containing slices:", len(tumor_slices))

print("\n===== TUMOR SIZE QUARTILE BOUNDARIES =====")
print(f"Small upper boundary (Q25): {q25:.2f} pixels")
print(f"Medium upper boundary (Q75): {q75:.2f} pixels")

print("\n===== TUMOR SIZE GROUP ANALYSIS =====")
print(size_results.to_string(index=False))

print("\n===== SLICE POSITION ANALYSIS =====")
print(position_results.to_string(index=False))

print("\n===== PATIENT TUMOR BURDEN VS DICE IMPROVEMENT =====")

print(
    "Total tumor pixels vs Dice improvement:"
)
print(
    f"Spearman rho: {total_burden_corr.statistic:.6f}"
)
print(
    f"P-value: {total_burden_corr.pvalue:.6g}"
)

print(
    "\nMean tumor-slice area vs Dice improvement:"
)
print(
    f"Spearman rho: {mean_slice_burden_corr.statistic:.6f}"
)
print(
    f"P-value: {mean_slice_burden_corr.pvalue:.6g}"
)

print("\n===== OUTPUT FILES =====")
print(OUTPUT_DIR)

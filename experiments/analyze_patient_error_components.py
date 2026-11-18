
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

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

OUTPUT_CSV = (
    PROJECT_ROOT / "results" / "patient_error_components.csv"
)


# ============================================================
# Reconstruct true-positive pixel counts
# ============================================================

def reconstruct_confusion_counts(df, model_name):

    gt = df["ground_truth_pixels"].to_numpy(dtype=np.float64)
    pred = df["predicted_pixels"].to_numpy(dtype=np.float64)
    dice = df["dice"].to_numpy(dtype=np.float64)

    # Dice = 2 * TP / (GT + Prediction)
    denominator = gt + pred

    tp_raw = np.zeros(len(df), dtype=np.float64)

    valid = denominator > 0

    tp_raw[valid] = (
        dice[valid] * denominator[valid] / 2.0
    )

    # TP must be an integer pixel count.
    tp = np.rint(tp_raw)

    if not np.allclose(tp_raw, tp, atol=1e-4):
        raise ValueError(
            f"{model_name}: reconstructed TP values "
            "are not sufficiently close to integers. "
            "Check the precision of the saved Dice values."
        )

    tp = tp.astype(np.int64)

    if np.any(tp < 0):
        raise ValueError(f"{model_name}: negative TP found.")

    if np.any(tp > gt):
        raise ValueError(f"{model_name}: TP exceeds GT.")

    if np.any(tp > pred):
        raise ValueError(f"{model_name}: TP exceeds prediction.")

    fn = gt.astype(np.int64) - tp
    fp = pred.astype(np.int64) - tp

    # Verify reconstructed Dice against original Dice.
    reconstructed_dice = np.ones(len(df), dtype=np.float64)

    reconstructed_dice[valid] = (
        2.0 * tp[valid] / denominator[valid]
    )

    if not np.allclose(
        reconstructed_dice,
        dice,
        atol=1e-5
    ):
        raise ValueError(
            f"{model_name}: reconstructed Dice "
            "does not match the original CSV."
        )

    return pd.DataFrame({
        "TP": tp,
        "FP": fp,
        "FN": fn,
    })


# ============================================================
# Load evaluation results
# ============================================================

baseline = pd.read_csv(BASELINE_CSV)
acu = pd.read_csv(ACU_CSV)

required_columns = {
    "ground_truth_pixels",
    "predicted_pixels",
    "dice",
}

for name, df in [
    ("Baseline", baseline),
    ("ACU-Net", acu)
]:

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} CSV is missing columns: {missing}"
        )

assert len(baseline) == len(acu), (
    "Baseline and ACU-Net CSV row counts differ."
)

assert np.array_equal(
    baseline["ground_truth_pixels"].to_numpy(),
    acu["ground_truth_pixels"].to_numpy()
), "Ground-truth pixel counts differ."


# ============================================================
# Reconstruct patient order
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
    "Patient slice mapping does not match CSV rows."
)

slice_patient_ids = []

for patient_id, count in zip(patient_ids, slice_counts):
    slice_patient_ids.extend([patient_id] * count)

baseline["patient_id"] = slice_patient_ids
acu["patient_id"] = slice_patient_ids


# ============================================================
# Reconstruct confusion components
# ============================================================

baseline_components = reconstruct_confusion_counts(
    baseline,
    "Baseline U-Net"
)

acu_components = reconstruct_confusion_counts(
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
# Aggregate pixel counts per patient
# ============================================================

baseline_patient = (
    baseline.groupby("patient_id")
    .agg(
        Baseline_TP=("TP", "sum"),
        Baseline_FP=("FP", "sum"),
        Baseline_FN=("FN", "sum"),
    )
)

acu_patient = (
    acu.groupby("patient_id")
    .agg(
        ACU_TP=("TP", "sum"),
        ACU_FP=("FP", "sum"),
        ACU_FN=("FN", "sum"),
    )
)

results = baseline_patient.join(acu_patient)


# ============================================================
# Calculate sensitivity and precision
# ============================================================

def safe_divide(numerator, denominator):

    return np.divide(
        numerator,
        denominator,
        out=np.full(len(numerator), np.nan, dtype=float),
        where=denominator != 0
    )


# Baseline metrics

results["Baseline_Sensitivity"] = safe_divide(
    results["Baseline_TP"].to_numpy(),
    (
        results["Baseline_TP"]
        + results["Baseline_FN"]
    ).to_numpy()
)

results["Baseline_Precision"] = safe_divide(
    results["Baseline_TP"].to_numpy(),
    (
        results["Baseline_TP"]
        + results["Baseline_FP"]
    ).to_numpy()
)

results["Baseline_FNR"] = safe_divide(
    results["Baseline_FN"].to_numpy(),
    (
        results["Baseline_TP"]
        + results["Baseline_FN"]
    ).to_numpy()
)


# ACU-Net metrics

results["ACU_Sensitivity"] = safe_divide(
    results["ACU_TP"].to_numpy(),
    (
        results["ACU_TP"]
        + results["ACU_FN"]
    ).to_numpy()
)

results["ACU_Precision"] = safe_divide(
    results["ACU_TP"].to_numpy(),
    (
        results["ACU_TP"]
        + results["ACU_FP"]
    ).to_numpy()
)

results["ACU_FNR"] = safe_divide(
    results["ACU_FN"].to_numpy(),
    (
        results["ACU_TP"]
        + results["ACU_FN"]
    ).to_numpy()
)


# ============================================================
# Calculate changes
# ============================================================

results["Sensitivity_Change"] = (
    results["ACU_Sensitivity"]
    - results["Baseline_Sensitivity"]
)

results["Precision_Change"] = (
    results["ACU_Precision"]
    - results["Baseline_Precision"]
)

results["FN_Change"] = (
    results["ACU_FN"]
    - results["Baseline_FN"]
)

results["FP_Change"] = (
    results["ACU_FP"]
    - results["Baseline_FP"]
)


# ============================================================
# Save patient-level results
# ============================================================

results.index.name = "Patient_ID"

results.to_csv(OUTPUT_CSV)


# ============================================================
# Aggregate summary
# ============================================================

print("\n" + "=" * 65)
print("QUANTITATIVE SEGMENTATION ERROR ANALYSIS")
print("=" * 65)

print("Validation patients:", len(results))


print("\n===== AGGREGATE PIXEL COUNTS =====")

for model in ["Baseline", "ACU"]:

    tp = int(results[f"{model}_TP"].sum())
    fp = int(results[f"{model}_FP"].sum())
    fn = int(results[f"{model}_FN"].sum())

    sensitivity = tp / (tp + fn)
    precision = tp / (tp + fp)
    fnr = fn / (tp + fn)

    print(f"\n{model}:")
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)
    print(f"Sensitivity: {sensitivity:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"False-negative rate: {fnr:.6f}")


print("\n===== PATIENT-LEVEL MEAN METRICS =====")

for metric in [
    "Baseline_Sensitivity",
    "ACU_Sensitivity",
    "Baseline_Precision",
    "ACU_Precision",
    "Baseline_FNR",
    "ACU_FNR",
]:

    print(
        f"{metric}: {results[metric].mean():.6f}"
    )


print("\n===== PATIENTS WITH INCREASED ERRORS =====")

print(
    "Patients with increased FN:",
    (results["FN_Change"] > 0).sum()
)

print(
    "Patients with reduced FN:",
    (results["FN_Change"] < 0).sum()
)

print(
    "Patients with increased FP:",
    (results["FP_Change"] > 0).sum()
)

print(
    "Patients with reduced FP:",
    (results["FP_Change"] < 0).sum()
)


print("\n===== SEVEN PREVIOUSLY WORSENED PATIENTS =====")

worsened_ids = [
    "Brats18_TCIA02_226_1",
    "Brats18_TCIA02_473_1",
    "Brats18_TCIA02_605_1",
    "Brats18_TCIA10_202_1",
    "Brats18_TCIA10_625_1",
    "Brats18_TCIA13_624_1",
    "Brats18_TCIA13_633_1",
]

print(
    results.loc[
        worsened_ids,
        [
            "Baseline_TP",
            "Baseline_FP",
            "Baseline_FN",
            "ACU_TP",
            "ACU_FP",
            "ACU_FN",
            "Sensitivity_Change",
            "Precision_Change",
            "FN_Change",
            "FP_Change",
        ]
    ].to_string()
)


print("\nResults saved to:")
print(OUTPUT_CSV)

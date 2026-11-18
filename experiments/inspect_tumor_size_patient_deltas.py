
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT = (
    PROJECT_ROOT
    / "results"
    / "tumor_size_analysis"
    / "tumor_slice_analysis.csv"
)

df = pd.read_csv(INPUT)

patient_group = (
    df.groupby(["patient_id", "Tumor_Size_Group"])
    .agg(
        Baseline_Dice=("Baseline_Dice", "mean"),
        ACU_Dice=("ACU_Dice", "mean"),
        Tumor_Slices=("Baseline_Dice", "size"),
    )
    .reset_index()
)

patient_group["Dice_Change"] = (
    patient_group["ACU_Dice"]
    - patient_group["Baseline_Dice"]
)

large = patient_group[
    patient_group["Tumor_Size_Group"] == "Large"
].copy()

print("\n===== LARGE-GROUP DELTA DISTRIBUTION =====")

print(large["Dice_Change"].describe())

print("\n===== FIVE LARGEST IMPROVEMENTS =====")

print(
    large.nlargest(5, "Dice_Change")[
        ["patient_id", "Tumor_Slices", "Dice_Change"]
    ].to_string(index=False)
)

print("\n===== FIVE LARGEST DETERIORATIONS =====")

print(
    large.nsmallest(5, "Dice_Change")[
        ["patient_id", "Tumor_Slices", "Dice_Change"]
    ].to_string(index=False)
)

print("\n===== PATIENTS WITH NEGATIVE IMPROVEMENT =====")

print(
    large[large["Dice_Change"] < 0][
        ["patient_id", "Tumor_Slices", "Dice_Change"]
    ].to_string(index=False)
)

# Bootstrap median improvement
rng = np.random.default_rng(42)

values = large["Dice_Change"].to_numpy()

bootstrap_medians = np.array([
    np.median(
        rng.choice(values, size=len(values), replace=True)
    )
    for _ in range(10000)
])

lower, upper = np.percentile(
    bootstrap_medians,
    [2.5, 97.5]
)

print("\n===== LARGE-GROUP MEDIAN BOOTSTRAP =====")
print(f"Median: {np.median(values):.6f}")
print(f"95% CI: [{lower:.6f}, {upper:.6f}]")


from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "tumor_size_analysis"
    / "tumor_slice_analysis.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "tumor_size_analysis"

OUTPUT_CSV = OUTPUT_DIR / "patient_level_size_validation.csv"

RNG = np.random.default_rng(42)
N_BOOTSTRAP = 10000

# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(INPUT_CSV)

required = {
    "patient_id",
    "Tumor_Size_Group",
    "Baseline_Dice",
    "ACU_Dice",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

print("Patients:", df["patient_id"].nunique())
print("Tumor-containing slices:", len(df))

# --------------------------------------------------
# Calculate patient-level means per size group
# --------------------------------------------------

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

# --------------------------------------------------
# Analyze each size group
# --------------------------------------------------

summary = []

for group in ["Small", "Medium", "Large"]:

    subset = patient_group[
        patient_group["Tumor_Size_Group"] == group
    ].copy()

    differences = subset["Dice_Change"].to_numpy()

    n = len(differences)

    if n == 0:
        continue

    improved = int((differences > 0).sum())
    worsened = int((differences < 0).sum())
    tied = int(np.isclose(differences, 0, atol=1e-12).sum())

    # Paired Wilcoxon signed-rank test
    if np.any(differences != 0):

        statistic, p_value = wilcoxon(
            differences,
            alternative="two-sided"
        )

    else:
        statistic = 0.0
        p_value = 1.0

    # Patient-level bootstrap
    bootstrap_means = np.empty(N_BOOTSTRAP)

    for i in range(N_BOOTSTRAP):

        sampled = RNG.choice(
            differences,
            size=n,
            replace=True
        )

        bootstrap_means[i] = sampled.mean()

    ci_lower, ci_upper = np.percentile(
        bootstrap_means,
        [2.5, 97.5]
    )

    summary.append({
        "Tumor_Size_Group": group,
        "Patients": n,
        "Improved": improved,
        "Worsened": worsened,
        "Tied": tied,
        "Baseline_Mean_Dice": subset["Baseline_Dice"].mean(),
        "ACU_Mean_Dice": subset["ACU_Dice"].mean(),
        "Mean_Dice_Change": differences.mean(),
        "Median_Dice_Change": np.median(differences),
        "Wilcoxon_Statistic": statistic,
        "Wilcoxon_P_Value": p_value,
        "Bootstrap_CI_Lower": ci_lower,
        "Bootstrap_CI_Upper": ci_upper,
    })

    print(f"\n===== {group.upper()} TUMOR GROUP =====")

    print("Patients:", n)
    print("Improved:", improved)
    print("Worsened:", worsened)
    print("Tied:", tied)

    print(
        f"Baseline mean Dice: "
        f"{subset['Baseline_Dice'].mean():.6f}"
    )

    print(
        f"ACU-Net mean Dice: "
        f"{subset['ACU_Dice'].mean():.6f}"
    )

    print(
        f"Mean patient Dice improvement: "
        f"{differences.mean():+.6f}"
    )

    print(
        f"Median patient Dice improvement: "
        f"{np.median(differences):+.6f}"
    )

    print("Wilcoxon statistic:", statistic)
    print("Wilcoxon p-value:", p_value)

    print("Bootstrap 95% CI:")
    print(f"[{ci_lower:.6f}, {ci_upper:.6f}]")

# --------------------------------------------------
# Save results
# --------------------------------------------------

summary_df = pd.DataFrame(summary)

summary_df.to_csv(OUTPUT_CSV, index=False)

print("\n===== FINAL SUMMARY =====")
print(summary_df.to_string(index=False))

print("\nResults saved to:")
print(OUTPUT_CSV)

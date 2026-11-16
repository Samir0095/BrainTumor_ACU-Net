import pandas as pd

path = "results/patient_level_comparison.csv"

df = pd.read_csv(path)

# Patients where ACU-Net performed worse
worsened = df[df["Dice_Improvement"] < 0].copy()

# Sort by largest degradation
worsened = worsened.sort_values("Dice_Improvement")

print("\n===== PATIENTS WHERE ACU-NET UNDERPERFORMED =====\n")

print(
    worsened[
        ["Baseline_Dice", "ACU_Net_Dice", "Dice_Improvement"]
    ].to_string()
)

print("\nNumber of worsened patients:", len(worsened))
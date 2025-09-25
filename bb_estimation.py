import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from matplotlib.ticker import PercentFormatter

df_original = pd.read_csv("data/dataproject2025.csv")
print(f"Number of rows with missing values: {df_original.isna().any(axis=1).sum()}")

df = pd.read_csv("data/dataproject2025_grad_boost_predictions_0925_2243.csv")

# Count rows where both 2nd and 3rd columns are 1
both_ones = df[(df['target'] == 1) & (df['grad_boost_predict'] == 1)]
both_zeros = df[(df['target'] == 0) & (df['grad_boost_predict'] == 0)]

false_positives = df[(df['target'] == 0) & (df['grad_boost_predict'] == 1)]
true_negatives = df[(df['target'] == 1) & (df['grad_boost_predict'] == 0)]

print(f"Number of both ones: {len(both_ones)}")
print(f"Number of both zeros: {len(both_zeros)}")
print(f"Number of false positives: {len(false_positives)}")
print(f"Number of true negatives: {len(true_negatives)}")

# 100% stacked bar chart: Correct vs Incorrect per target class
target0 = df['target'] == 0
target1 = df['target'] == 1
pred0 = df['grad_boost_predict'] == 0
pred1 = df['grad_boost_predict'] == 1

correct0 = int((target0 & pred0).sum())
incorrect0 = int((target0 & pred1).sum())
correct1 = int((target1 & pred1).sum())
incorrect1 = int((target1 & pred0).sum())

tot0 = correct0 + incorrect0
tot1 = correct1 + incorrect1

corr_pct = [ (correct0 / tot0) if tot0 else 0.0, (correct1 / tot1) if tot1 else 0.0 ]
incorr_pct = [ (incorrect0 / tot0) if tot0 else 0.0, (incorrect1 / tot1) if tot1 else 0.0 ]

categories = ['0', '1']

fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(categories, corr_pct, label='Correct', color='#2ca02c')
ax.bar(categories, incorr_pct, bottom=corr_pct, label='Incorrect', color='#d62728')

ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_xlabel('Target class')
ax.set_ylabel('Percent of samples')
ax.set_title('Correct vs Incorrect by Target (Normalized)')
ax.legend()

# Annotate bars with counts and percentages
for i in range(len(categories)):
    if corr_pct[i] > 0:
        count = correct0 if i == 0 else correct1
        ax.text(i, corr_pct[i] / 2, f"{count} ({corr_pct[i]*100:.1f}%)", ha='center', va='center', color='white', fontsize=10, fontweight='bold')
    if incorr_pct[i] > 0:
        count = incorrect0 if i == 0 else incorrect1
        base = corr_pct[i]
        ax.text(i, base + incorr_pct[i] / 2, f"{count} ({incorr_pct[i]*100:.1f}%)", ha='center', va='center', color='white', fontsize=10, fontweight='bold')

os.makedirs("outputs", exist_ok=True)
out_path = f"outputs/accuracy_by_target_stacked_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"Saved plot to {out_path}")
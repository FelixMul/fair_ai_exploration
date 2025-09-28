import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import roc_curve, auc

df_original = pd.read_csv("data/dataproject2025.csv")
print(f"Number of rows with missing values: {df_original.isna().any(axis=1).sum()}")

df = pd.read_csv("data/dataproject2025_grad_boost_predictions_0928_1753.csv")

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

# --- ROC Curve and AUC ---
roc_y_true = df['target'].values
score_column = None
if 'grad_boost_proba' in df.columns:
    score_column = 'grad_boost_proba'
elif 'grad_boost_score_raw' in df.columns:
    score_column = 'grad_boost_score_raw'
else:
    score_column = None

timestamp = datetime.now().strftime('%Y%m%d_%H%M')
if score_column is not None:
    scores = df[score_column].values
    fpr, tpr, thresholds = roc_curve(roc_y_true, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    roc_path = f"outputs/roc_curve_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"AUC: {roc_auc:.4f}")
    print(f"Saved ROC curve to {roc_path}")
else:
    # Fallback: attempt ROC using binary predictions if no scores exist
    if 'grad_boost_predict' in df.columns:
        preds = df['grad_boost_predict'].values
        # With binary-only predictions the ROC curve has at most two points and AUC reduces to accuracy for balanced classes; inform user.
        fpr, tpr, thresholds = roc_curve(roc_y_true, preds)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC ~ {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Chance')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (binary-only fallback)')
        plt.legend(loc="lower right")
        roc_path = f"outputs/roc_curve_binary_fallback_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(roc_path, dpi=150)
        plt.close()
        print("Warning: No score/probability column found. Using binary predictions for ROC; AUC is less informative.")
        print(f"Fallback AUC (binary): {roc_auc:.4f}")
        print(f"Saved ROC curve (fallback) to {roc_path}")
    else:
        print("Warning: Could not compute ROC/AUC because no score or prediction columns were found.")
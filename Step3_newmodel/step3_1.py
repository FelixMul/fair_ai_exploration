# =====================================
# Step 3 (FULL) — Merge + Subgroup analysis (white background, navy text)
# Repo-friendly: relative paths, same file names as your local version
# - Reads: data/dataproject2025.csv
#          data/dataproject2025_grad_boost_predictions_0925_2243.csv
# - Writes: outputs_step3_full/*.png, *.csv
# =====================================

from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# -------------------------------------------------------------
# Part A — Paths & Merge
# -------------------------------------------------------------
# Find repo root: current dir; if no ./data, try parent
HERE = Path(__file__).resolve().parent
if (HERE / "data").exists():
    ROOT = HERE
elif (HERE.parent / "data").exists():
    ROOT = HERE.parent
else:
    raise FileNotFoundError("Cannot locate 'data/' folder. Put this script inside the repo or its subfolder.")

DATA = ROOT / "data"
OUT = ROOT / "outputs_step3_full"
OUT.mkdir(exist_ok=True)

# Keep file names unchanged (as requested)
data_path = DATA / "dataproject2025.csv"
pred_path = DATA / "dataproject2025_grad_boost_predictions_0925_2243.csv"

# Load
df_orig = pd.read_csv(data_path)
df_pred = pd.read_csv(pred_path)
print("Original dataset shape:", df_orig.shape)
print("Prediction dataset shape:", df_pred.shape)

# Merge on identifier "Unnamed: 0"
if "Unnamed: 0" not in df_orig.columns or "Unnamed: 0" not in df_pred.columns:
    raise KeyError("Identifier 'Unnamed: 0' missing in one of the files.")
df_merged = df_orig.merge(df_pred, on="Unnamed: 0", how="inner")
print("Merged dataset shape:", df_merged.shape)

# Save small sample for sanity check
df_merged.head(20).to_csv(OUT / "03_merged_head20.csv", index=False)

# -------------------------------------------------------------
# Part B — Subgroup accuracy plots (white background, navy text)
# -------------------------------------------------------------

# Normalize columns to lower snake-case style
df_merged.columns = [c.strip().lower() for c in df_merged.columns]
print("Available columns:", df_merged.columns.tolist())

# Auto-detect target and prediction columns (keep your rule)
target_col_candidates = [c for c in df_merged.columns if "target" in c]
pred_col_candidates   = [c for c in df_merged.columns if "grad_boost" in c]

if not target_col_candidates:
    raise KeyError("Cannot find a 'target' column in merged data.")
if not pred_col_candidates:
    raise KeyError("Cannot find a 'grad_boost' prediction column in merged data.")

target_col = target_col_candidates[0]
pred_col   = pred_col_candidates[0]
print(f"Using target column: {target_col}, prediction column: {pred_col}")

# Helper: consistent white background + navy text styling
def style_axes(ax, title, xlab, ylab):
    """Apply white background and navy text for a seaborn/matplotlib Axes."""
    ax.set_title(title, color="navy")
    ax.set_xlabel(xlab, color="navy")
    ax.set_ylabel(ylab, color="navy")
    ax.tick_params(colors="navy")
    for spine in ax.spines.values():
        spine.set_edgecolor("navy")

# Helper: subgroup accuracy DataFrame
def subgroup_accuracy(df, group_col, out_name="accuracy"):
    grp = (
        df.dropna(subset=[group_col])
          .groupby(group_col)
          .apply(lambda g: accuracy_score(g[target_col], g[pred_col]))
          .reset_index(name=out_name)
          .sort_values(group_col)
    )
    return grp

# 1) By Year (from issue_d)
if "issue_d" in df_merged.columns:
    issue_parsed = pd.to_datetime(df_merged["issue_d"], errors="coerce", infer_datetime_format=True)
    if issue_parsed.notna().any():
        df_merged["issue_year"] = issue_parsed.dt.year
    else:
        df_merged["issue_year"] = np.floor(pd.to_numeric(df_merged["issue_d"], errors="coerce"))
    grp_year = subgroup_accuracy(df_merged, "issue_year")
    grp_year.to_csv(OUT / "03_subgroup_year.csv", index=False)

    plt.style.use("default")
    fig = plt.figure(figsize=(7, 4), facecolor="white")
    ax = sns.barplot(data=grp_year, x="issue_year", y="accuracy", palette="Blues_d")
    style_axes(ax, "Subgroup Accuracy by Year", "Year", "Accuracy")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "03_subgroup_year.png", dpi=150, facecolor="white")
    plt.close(fig)

# 2) By Grade
if "grade" in df_merged.columns:
    grp_grade = subgroup_accuracy(df_merged, "grade")
    grp_grade.to_csv(OUT / "03_subgroup_grade.csv", index=False)

    plt.style.use("default")
    fig = plt.figure(figsize=(6, 4), facecolor="white")
    ax = sns.barplot(data=grp_grade, x="grade", y="accuracy", palette="Blues_d")
    style_axes(ax, "Subgroup Accuracy by Grade", "Grade", "Accuracy")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "03_subgroup_grade.png", dpi=150, facecolor="white")
    plt.close(fig)

# 3) By Sub-grade
if "sub_grade" in df_merged.columns:
    grp_sub = subgroup_accuracy(df_merged, "sub_grade")
    grp_sub.to_csv(OUT / "03_subgroup_subgrade.csv", index=False)

    plt.style.use("default")
    fig = plt.figure(figsize=(10, 4), facecolor="white")
    ax = sns.barplot(data=grp_sub, x="sub_grade", y="accuracy", palette="Blues_d")
    style_axes(ax, "Subgroup Accuracy by Sub-grade", "Sub-grade", "Accuracy")
    ax.set_ylim(0, 1)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_color("navy")
    fig.tight_layout()
    fig.savefig(OUT / "03_subgroup_subgrade.png", dpi=150, facecolor="white")
    plt.close(fig)

# 4) By Income Quartile
if "annual_inc" in df_merged.columns:
    # Robust to ties: try qcut; fallback to cut
    try:
        df_merged["income_bucket"] = pd.qcut(df_merged["annual_inc"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    except ValueError:
        df_merged["income_bucket"] = pd.cut(df_merged["annual_inc"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    grp_income = subgroup_accuracy(df_merged, "income_bucket")
    grp_income.to_csv(OUT / "03_subgroup_income.csv", index=False)

    plt.style.use("default")
    fig = plt.figure(figsize=(6, 4), facecolor="white")
    ax = sns.barplot(data=grp_income, x="income_bucket", y="accuracy", palette="Blues_d")
    style_axes(ax, "Subgroup Accuracy by Income Quartile", "Income Quartile", "Accuracy")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "03_subgroup_income.png", dpi=150, facecolor="white")
    plt.close(fig)

# 5) By Loan Duration
if "loan duration" in df_merged.columns:
    grp_ld = subgroup_accuracy(df_merged, "loan duration")
    grp_ld.to_csv(OUT / "03_subgroup_loanduration.csv", index=False)

    plt.style.use("default")
    fig = plt.figure(figsize=(6, 4), facecolor="white")
    ax = sns.barplot(data=grp_ld, x="loan duration", y="accuracy", palette="Blues_d")
    style_axes(ax, "Subgroup Accuracy by Loan Duration", "Loan Duration", "Accuracy")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "03_subgroup_loanduration.png", dpi=150, facecolor="white")
    plt.close(fig)

print("✅ Extended Subgroup Analysis done. Results saved in:", OUT)

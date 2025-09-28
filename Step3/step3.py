# ==============================
# STEP 3 (FULL): performance + structural stability
# - Merges original dataset with predictions by ID
# - Overall metrics + CM + ROC/PR (prob fallback)
# - Bootstrap (Accuracy/Precision/Recall/F1)
# - Subgroups: year, grade, sub_grade, income quartiles, loan duration
# - Saves to ./outputs/step3_full
# ==============================

from pathlib import Path
import glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
)

# ---------- Paths ----------
BASE = Path.cwd()        # run from repo root
DATA = BASE / "data"
OUT  = BASE / "outputs" / "step3_full"
OUT.mkdir(parents=True, exist_ok=True)

RAW_CSV  = DATA / "dataproject2025.csv"
pred_candidates = sorted(glob.glob(str(DATA / "dataproject2025_grad_boost_predictions*.csv")))
assert RAW_CSV.exists(), "dataproject2025.csv not found under ./data."
assert pred_candidates, "No predictions CSV found under ./data with pattern 'dataproject2025_grad_boost_predictions*.csv'."
PRED_CSV = Path(pred_candidates[-1])
print("Using predictions file:", PRED_CSV.name)

# ---------- Load & normalize ----------
df_raw = pd.read_csv(RAW_CSV)
df_pred = pd.read_csv(PRED_CSV)

df_raw.columns  = [c.strip().lower() for c in df_raw.columns]
df_pred.columns = [c.strip().lower() for c in df_pred.columns]

# Helpers
def pick_col(df, preferred, contains_any=None, exclude_contains=None):
    for c in preferred:
        if c in df.columns:
            return c
    if contains_any:
        for c in df.columns:
            if any(k in c for k in contains_any):
                if not exclude_contains or all(k not in c for k in (exclude_contains or [])):
                    return c
    return None

# Detect ID column for merge
id_raw  = pick_col(df_raw,  ["unnamed: 0","id","row_id"], contains_any=["unnamed: 0","id","row_id"])
id_pred = pick_col(df_pred, ["unnamed: 0","id","row_id"], contains_any=["unnamed: 0","id","row_id"])
assert id_raw and id_pred, f"Cannot find a common ID. Raw: {df_raw.columns.tolist()} | Pred: {df_pred.columns.tolist()}"

# Detect target/pred columns in predictions file
target_pred_file = pick_col(df_pred, ["target"], contains_any=["target"])
pred_label_col   = pick_col(df_pred, ["grad_boost_predict","predictions","y_pred"], contains_any=["predict","pred"], exclude_contains=["prob"])
assert target_pred_file and pred_label_col, "Cannot locate target/pred columns in prediction file."

# Merge
df_pred_ren = df_pred[[id_pred, target_pred_file, pred_label_col]].rename(
    columns={id_pred:"row_id", target_pred_file:"target_pred", pred_label_col:"pred_label"}
)
df_raw_ren = df_raw.copy(); df_raw_ren["row_id"] = df_raw_ren[id_raw]
dfm = df_raw_ren.merge(df_pred_ren, on="row_id", how="inner")

# Choose ground-truth
target_col = "target" if "target" in dfm.columns else "target_pred"
assert target_col in dfm.columns, "No ground-truth target found after merge."
dfm[target_col] = dfm[target_col].astype(int)
dfm["pred_label"] = dfm["pred_label"].astype(int)

y_true = dfm[target_col]
y_pred = dfm["pred_label"]

# ---------- Overall metrics ----------
metrics = {
    "n_samples":          int(len(y_true)),
    "accuracy":           float(accuracy_score(y_true, y_pred)),
    "precision":          float(precision_score(y_true, y_pred, zero_division=0)),
    "recall":             float(recall_score(y_true, y_pred, zero_division=0)),
    "f1":                 float(f1_score(y_true, y_pred, zero_division=0)),
    "balanced_accuracy":  float(balanced_accuracy_score(y_true, y_pred)),
    "id_key":             "row_id",
    "target_col":         target_col,
    "pred_label_col":     "pred_label",
    "pred_file":          PRED_CSV.name
}
(Path(OUT/"03_overall_metrics.json")).write_text(json.dumps(metrics, indent=2))
pd.DataFrame([metrics]).to_csv(OUT/"03_overall_metrics.csv", index=False)
print(json.dumps(metrics, indent=2))

# ---------- Confusion Matrix ----------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred 0","Pred 1"], yticklabels=["True 0","True 1"])
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT/"03_confusion_matrix.png", dpi=150)
plt.close()

# ---------- ROC & PR (probability fallback) ----------
# Try to find probability column in either raw or predictions file
prob_col = pick_col(dfm, ["predicted probabilities","y_proba","prob","probability"], contains_any=["prob"])
if prob_col is None:
    # fallback to hard labels; ROC/PR become stepwise and AUC/AP less informative
    y_prob = y_pred.values
else:
    y_prob = pd.to_numeric(dfm[prob_col], errors="coerce").fillna(0).values

fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],"--",color="grey")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve"); plt.legend(); plt.tight_layout()
plt.savefig(OUT/"03_roc_curve.png", dpi=150); plt.close()

prec, rec, _ = precision_recall_curve(y_true, y_prob)
ap = average_precision_score(y_true, y_prob)
plt.figure(figsize=(6,5))
plt.plot(rec, prec, lw=2, color="darkorange", label=f"AP = {ap:.2f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curve"); plt.legend(); plt.tight_layout()
plt.savefig(OUT/"03_pr_curve.png", dpi=150); plt.close()

# ---------- Bootstrap stability (4 metrics) ----------
B = 200
rng = np.random.default_rng(42)
boot = {"accuracy":[], "precision":[], "recall":[], "f1":[]}
n = len(y_true)
for _ in range(B):
    idx = rng.integers(0, n, size=n)
    yt, yp = y_true.iloc[idx], y_pred.iloc[idx]
    boot["accuracy"].append(accuracy_score(yt, yp))
    boot["precision"].append(precision_score(yt, yp, zero_division=0))
    boot["recall"].append(recall_score(yt, yp, zero_division=0))
    boot["f1"].append(f1_score(yt, yp, zero_division=0))

boot_df = pd.DataFrame(boot)
boot_df.describe(percentiles=[0.025,0.5,0.975]).to_csv(OUT/"03_bootstrap_summary.csv")

plt.figure(figsize=(8,5))
boot_df.boxplot()
plt.title("Bootstrap Distribution of Metrics (B=200)")
plt.ylabel("Score"); plt.tight_layout()
plt.savefig(OUT/"03_bootstrap_boxplot.png", dpi=150); plt.close()

# ---------- Subgroup analysis ----------
def subgroup_accuracy(df, group_col):
    g = (df.dropna(subset=[group_col])
           .groupby(group_col)
           .apply(lambda t: accuracy_score(t[target_col], t["pred_label"]))
           .reset_index(name="accuracy")
           .sort_values(group_col))
    return g

# Year (parse issue_d)
if "issue_d" in dfm.columns:
    parsed = pd.to_datetime(dfm["issue_d"], errors="coerce", infer_datetime_format=True)
    dfm["issue_year"] = parsed.dt.year.where(parsed.notna(),
                        np.floor(pd.to_numeric(dfm["issue_d"], errors="coerce")))
    if dfm["issue_year"].notna().any():
        g_year = subgroup_accuracy(dfm, "issue_year")
        g_year.to_csv(OUT/"03_subgroup_year.csv", index=False)
        plt.figure(figsize=(8,4))
        sns.barplot(data=g_year, x="issue_year", y="accuracy", palette="Blues_d")
        plt.ylim(0,1); plt.title("Accuracy by Year"); plt.tight_layout()
        plt.savefig(OUT/"03_subgroup_year.png", dpi=150); plt.close()

# Grade
if "grade" in dfm.columns:
    g_grade = subgroup_accuracy(dfm, "grade")
    g_grade.to_csv(OUT/"03_subgroup_grade.csv", index=False)
    plt.figure(figsize=(6,4))
    sns.barplot(data=g_grade, x="grade", y="accuracy", palette="viridis")
    plt.ylim(0,1); plt.title("Accuracy by Grade"); plt.tight_layout()
    plt.savefig(OUT/"03_subgroup_grade.png", dpi=150); plt.close()

# Sub-grade
if "sub_grade" in dfm.columns:
    g_sub = subgroup_accuracy(dfm, "sub_grade")
    g_sub.to_csv(OUT/"03_subgroup_subgrade.csv", index=False)
    plt.figure(figsize=(10,4))
    sns.barplot(data=g_sub, x="sub_grade", y="accuracy", palette="coolwarm")
    plt.ylim(0,1); plt.xticks(rotation=45)
    plt.title("Accuracy by Sub-grade"); plt.tight_layout()
    plt.savefig(OUT/"03_subgroup_subgrade.png", dpi=150); plt.close()

# Income quartiles
if "annual_inc" in dfm.columns:
    # handle edge cases with many ties
    try:
        dfm["income_bucket"] = pd.qcut(dfm["annual_inc"], 4, labels=["Q1","Q2","Q3","Q4"])
    except ValueError:
        dfm["income_bucket"] = pd.cut(dfm["annual_inc"], bins=4, labels=["Q1","Q2","Q3","Q4"])
    g_inc = subgroup_accuracy(dfm, "income_bucket")
    g_inc.to_csv(OUT/"03_subgroup_income.csv", index=False)
    plt.figure(figsize=(6,4))
    sns.barplot(data=g_inc, x="income_bucket", y="accuracy", palette="magma")
    plt.ylim(0,1); plt.title("Accuracy by Income Quartile"); plt.tight_layout()
    plt.savefig(OUT/"03_subgroup_income.png", dpi=150); plt.close()

# Loan duration
if "loan duration" in dfm.columns:
    g_ld = subgroup_accuracy(dfm, "loan duration")
    g_ld.to_csv(OUT/"03_subgroup_loanduration.csv", index=False)
    plt.figure(figsize=(6,4))
    sns.barplot(data=g_ld, x="loan duration", y="accuracy", palette="Set2")
    plt.ylim(0,1); plt.title("Accuracy by Loan Duration"); plt.tight_layout()
    plt.savefig(OUT/"03_subgroup_loanduration.png", dpi=150); plt.close()

print("✅ Step3 FULL done. Check outputs in:", OUT)

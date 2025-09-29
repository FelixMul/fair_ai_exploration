# ===========================================
# Step 4 (Approach A): Use ONLY coworker's prediction CSV as teacher
# - Rename all non-id columns from prediction CSV with suffix "__PRED"
# - Pick y_surr strictly from "__PRED" columns (prefer probabilities, fallback to hard labels)
# - Train global surrogates: Decision Tree (depth=3) & Lasso
# - Save metrics, plots, textual rules, and Step1 vs Step4 R^2 comparison
# ===========================================

import re, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from packaging.version import parse as vparse
from sklearn import __version__ as skver
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------
# Paths
# -----------------------------
DL = Path.home() / "Downloads"
RAW = DL / "dataproject2025.csv"
PRED = DL / "dataproject2025_grad_boost_predictions_0925_2243.csv"  # coworker's predictions
OUT = DL / "outputs_step4_csv"
OUT.mkdir(parents=True, exist_ok=True)

print("Raw CSV:", RAW)
print("Pred CSV:", PRED)
print("Outputs ->", OUT)

# -----------------------------
# 0) Load data
# -----------------------------
df = pd.read_csv(RAW)
pred = pd.read_csv(PRED)

# -----------------------------
# 1) Rename prediction columns with __PRED suffix and merge
#    - This avoids name collisions with original df columns
#    - We ONLY search y_surr among the suffixed columns
# -----------------------------
id_keys = [c for c in ["Unnamed: 0", "id", "ID", "index"] if c in df.columns and c in pred.columns]
pred_renamed = pred.copy()
non_id_cols = [c for c in pred_renamed.columns if c not in id_keys]
pred_renamed = pred_renamed.rename(columns={c: f"{c}__PRED" for c in non_id_cols})

if id_keys:
    key = id_keys[0]
    dfm = df.merge(pred_renamed, on=key, how="inner")
else:
    # Fallback: same order alignment
    dfm = df.copy()
    for c in pred_renamed.columns:
        if c not in dfm.columns:
            dfm[c] = pred_renamed[c].values

print("Merged shape:", dfm.shape)

# -----------------------------
# 2) Pick teacher target y_surr STRICTLY from __PRED columns
#    - Prefer probabilities (regex 'proba|prob|probabil')
#    - Fallback to hard labels (look for 'predict' or explicit grad_boost_predict__PRED)
# -----------------------------
pred_side_cols = [c for c in dfm.columns if c.endswith("__PRED")]

# Prefer probability-like columns
proba_candidates = [c for c in pred_side_cols if re.search(r"proba|prob|probabil", c, re.I)]
y_col = None
use_hard = False

for c in proba_candidates:
    if pd.api.types.is_numeric_dtype(dfm[c]):
        y_col = c
        break

# Fallback to labels (hard 0/1)
if y_col is None:
    label_candidates = [c for c in pred_side_cols if re.search(r"(?:^|_)predict(?:ion)?__PRED$|label__PRED$", c, re.I)]
    # Also try the common name from your coworker's script
    if "grad_boost_predict__PRED" in pred_side_cols:
        label_candidates = ["grad_boost_predict__PRED"] + label_candidates
    for c in label_candidates:
        if c in dfm.columns and pd.api.types.is_numeric_dtype(dfm[c]):
            y_col = c
            use_hard = True
            break

assert y_col is not None, "No usable y_surr found in prediction CSV columns (with __PRED suffix)."
print(f"Teacher y_surr column (from PRED): {y_col} | Type: {'HARD labels' if use_hard else 'PROBABILITIES'}")

y_raw = pd.to_numeric(dfm[y_col], errors="coerce")
mask = y_raw.notna() & np.isfinite(y_raw)
y = y_raw[mask].astype(float)

# (Optional) Debug: show correlation with Step 1's 'Predicted probabilities' if it exists
if "Predicted probabilities" in dfm.columns:
    step1_y = pd.to_numeric(dfm["Predicted probabilities"], errors="coerce")
    corr = np.corrcoef(y, step1_y[mask])[0, 1]
    same = np.allclose(y.values, step1_y[mask].values, equal_nan=True)
    print(f"[Debug] Corr( Step4_y , Step1 'Predicted probabilities') = {corr:.6f} | Exactly same? {same}")

# -----------------------------
# 3) Build X for surrogates (avoid leakage)
#    - Remove target & any columns from prediction side (all __PRED)
# -----------------------------
leak = {
    "target", "Predictions", "Predicted probabilities", "DP",
    "Unnamed: 0", "issue_d"
}
# Exclude ALL pred-side columns to avoid leakage
leak.update(pred_side_cols)
# Also exclude the chosen y_col explicitly (already included by pred_side_cols)
leak.add(y_col)

X_cols = [c for c in dfm.columns if c not in leak]
X_raw = dfm.loc[mask, X_cols].copy().replace([np.inf, -np.inf], np.nan)

num_feat = [c for c in X_raw.columns if pd.api.types.is_numeric_dtype(X_raw[c])]
cat_feat = [c for c in X_raw.columns if pd.api.types.is_object_dtype(X_raw[c])]

# Optional: drop extremely high-cardinality free-text
if "emp_title" in cat_feat and X_raw["emp_title"].nunique() > 5000:
    X_raw.drop(columns=["emp_title"], inplace=True)
    cat_feat.remove("emp_title")

# -----------------------------
# 4) Preprocessor (OneHotEncoder API compatibility across versions)
# -----------------------------
ohe_kwargs = {"handle_unknown": "ignore"}
if vparse(skver) >= vparse("1.2"):
    ohe_kwargs["sparse_output"] = True
else:
    ohe_kwargs["sparse"] = True

transformers = []
if num_feat:
    transformers.append(("num", SimpleImputer(strategy="median"), num_feat))
if cat_feat:
    transformers.append(("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(**ohe_kwargs))
    ]), cat_feat))
pre = ColumnTransformer(transformers)

# -----------------------------
# 5) Fit surrogates
# -----------------------------
# Surrogate 1: shallow decision tree
tree = Pipeline([
    ("pre", pre),
    ("model", DecisionTreeRegressor(max_depth=3, random_state=42))
])
tree.fit(X_raw, y)
yhat_tree = tree.predict(X_raw)
m_tree = {"R2": float(r2_score(y, yhat_tree)),
          "MSE": float(mean_squared_error(y, yhat_tree))}
(Path(OUT/"04_surrogate_tree_metrics.json")).write_text(json.dumps(m_tree, indent=2))

# Surrogate 2: Lasso
lasso = Pipeline([
    ("pre", pre),
    ("model", LassoCV(cv=5, random_state=42, n_jobs=-1))
])
lasso.fit(X_raw, y)
yhat_lasso = lasso.predict(X_raw)
m_lasso = {"R2": float(r2_score(y, yhat_lasso)),
           "MSE": float(mean_squared_error(y, yhat_lasso))}
(Path(OUT/"04_surrogate_lasso_metrics.json")).write_text(json.dumps(m_lasso, indent=2))

print("Tree R2=%.3f | Lasso R2=%.3f" % (m_tree["R2"], m_lasso["R2"]))

# -----------------------------
# 6) Feature names for plotting
# -----------------------------
pre_for_names = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_feat) if num_feat else ("num","drop",[]),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(**ohe_kwargs))]), cat_feat) if cat_feat else ("cat","drop",[])
    ]
)
pre_for_names.fit(X_raw)

feature_names = []
if num_feat:
    feature_names += list(num_feat)
if cat_feat:
    ohe_fitted = pre_for_names.named_transformers_["cat"].named_steps["ohe"]
    if hasattr(ohe_fitted, "get_feature_names_out"):
        cat_names = ohe_fitted.get_feature_names_out(cat_feat).tolist()
    else:
        cat_names = ohe_fitted.get_feature_names(cat_feat).tolist()
    feature_names += cat_names

# -----------------------------
# 7) Plots for slides
# -----------------------------
# (a) Decision Tree structure + textual rules
fig, ax = plt.subplots(figsize=(12, 6))
model_dt = tree.named_steps["model"]
names_ok = feature_names if hasattr(model_dt, "n_features_") and len(feature_names) == model_dt.n_features_ else None
plot_tree(model_dt, feature_names=names_ok, filled=True, rounded=True, max_depth=3, ax=ax)
plt.title(f"Step 4 Surrogate — Decision Tree (depth=3)\nTeacher: {'HARD' if use_hard else 'PROBA'} from {PRED.name} (only __PRED columns)")
plt.tight_layout(); plt.savefig(OUT/"04_tree_structure.png", dpi=160); plt.close(fig)

rules_txt = export_text(model_dt, feature_names=(feature_names if names_ok else None))
(Path(OUT/"04_tree_rules.txt")).write_text(rules_txt)

# (b) Lasso top coefficients
coef_vec = lasso.named_steps["model"].coef_
feat_for_coef = feature_names if len(feature_names) == len(coef_vec) else [f"f_{i}" for i in range(len(coef_vec))]
coef_df = pd.DataFrame({"feature": feat_for_coef, "coef": coef_vec})
nz = coef_df[coef_df["coef"] != 0]
if nz.empty:
    nz = coef_df.copy()
topk = nz.reindex(nz["coef"].abs().sort_values(ascending=False).index).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(topk["feature"][::-1], topk["coef"][::-1])
ax.set_xlabel("Coefficient"); ax.set_ylabel("Feature")
ax.set_title(f"Step 4 Surrogate — Top Lasso Coefficients\nTeacher: {'HARD' if use_hard else 'PROBA'} from {PRED.name} (only __PRED columns)")
plt.tight_layout(); plt.savefig(OUT/"04_lasso_topcoef.png", dpi=160); plt.close(fig)

# -----------------------------
# 8) Compare with Step 1 (fixed paths you used)
# -----------------------------
step1_tree_path = Path("/Users/xujialong/Downloads/outputs/01_surrogate_tree_metrics.json")
step1_lasso_path = Path("/Users/xujialong/Downloads/outputs/01_surrogate_lasso_metrics.json")

def try_load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

step1_tree = try_load_json(step1_tree_path)
step1_lasso = try_load_json(step1_lasso_path)

cmp = {
    "teacher_source": f"{'HARD' if use_hard else 'PROBA'}:{PRED.name}",
    "y_column_used": y_col,
    "step1_tree_R2": (step1_tree or {}).get("R2", None),
    "step1_lasso_R2": (step1_lasso or {}).get("R2", None),
    "step4_tree_R2": m_tree["R2"],
    "step4_lasso_R2": m_lasso["R2"]
}
(Path(OUT / "04_step1_vs_step4_R2.json")).write_text(json.dumps(cmp, indent=2))

print("✅ Step 4 (Approach A) finished. See:", OUT)

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score, mean_squared_error

# --- Directories relative to repo root ---
CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR.parent / "data"
OUT_DIR = CURRENT_DIR.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# --- Load dataset ---
csv_path = DATA_DIR / "dataproject2025.csv"
df = pd.read_csv(csv_path)

# --- Surrogate target ---
y_surr_col = None
for cand in ["Predicted probabilities", "DP", "Predictions"]:
    if cand in df.columns:
        y_surr_col = cand
        break
assert y_surr_col, "No surrogate target column found"

y_surr = pd.to_numeric(df[y_surr_col], errors="coerce")
mask = y_surr.notna() & np.isfinite(y_surr)
y_surr = y_surr[mask]

# --- Features ---
drop_cols = {"target","Predictions","Predicted probabilities","DP"}
X = df.loc[mask, [c for c in df.columns if c not in drop_cols and c != "issue_d"]]

num_feat = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_feat = [c for c in X.columns if pd.api.types.is_object_dtype(X[c])]

if "emp_title" in cat_feat and X["emp_title"].nunique() > 5000:
    X = X.drop(columns=["emp_title"])
    cat_feat.remove("emp_title")

# --- Preprocessor ---
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_feat),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=True))
    ]), cat_feat)
])

# --- Decision Tree surrogate ---
tree = Pipeline([
    ("pre", pre),
    ("model", DecisionTreeRegressor(max_depth=3, random_state=42))
])
tree.fit(X, y_surr)
y_hat_tree = tree.predict(X)
metrics_tree = {"R2": r2_score(y_surr, y_hat_tree), "MSE": mean_squared_error(y_surr, y_hat_tree)}

# --- Lasso surrogate ---
lasso = Pipeline([
    ("pre", pre),
    ("model", LassoCV(cv=5, random_state=42, n_jobs=-1))
])
lasso.fit(X, y_surr)
y_hat_lasso = lasso.predict(X)
metrics_lasso = {"R2": r2_score(y_surr, y_hat_lasso), "MSE": mean_squared_error(y_surr, y_hat_lasso)}

# --- Save ---
(OUT_DIR / "step1_tree_metrics.json").write_text(json.dumps(metrics_tree, indent=2))
(OUT_DIR / "step1_lasso_metrics.json").write_text(json.dumps(metrics_lasso, indent=2))

print("✅ Step1 done")
print("Tree:", metrics_tree)
print("Lasso:", metrics_lasso)

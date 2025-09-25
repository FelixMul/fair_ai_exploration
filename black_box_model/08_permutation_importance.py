#!/usr/bin/env python
# coding: utf-8

# In[19]:


import os, numpy as np, pandas as pd
from datetime import datetime
from collections import OrderedDict

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

# 1) Charger données + CSV de Félix
df = pd.read_csv("dataproject2025.csv")
preds = pd.read_csv("dataproject2025_grad_boost_predictions_0924_1031.csv")
if "Unnamed: 0" not in df.columns:
    df["Unnamed: 0"] = np.arange(len(df))
df = df.merge(preds[["Unnamed: 0","grad_boost_predict"]], on="Unnamed: 0", how="left")
df["grad_boost_predict"] = df["grad_boost_predict"].astype(int)

# 2) Définir y et X SANS colonnes interdites (pas de fuite)
cols_excl_base = ['target','Predictions','Predicted probabilities','Unnamed: 0']
cols_leak = ['grad_boost_predict','DP','estimated_default_probability','default_probability','proba_default']
y = df['target'].astype(int)
X = df.drop(columns=[c for c in cols_excl_base+cols_leak if c in df.columns], errors='ignore')

# 3) Split identique à Félix
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4) Cible de l’émulateur = sorties de Félix (mais PAS dans X)
y_felix = df.loc[X.index, "grad_boost_predict"].astype(int)
y_felix_train = y_felix.loc[X_train.index]
y_felix_test  = y_felix.loc[X_test.index]

# 5) Pipeline préproc identique + émulateur
num_features = X.select_dtypes(include=np.number).columns.tolist()
cat_features = X.select_dtypes(include='object').columns.tolist()

numeric_tr = Pipeline([('imputer', SimpleImputer(strategy='median')),
                       ('scaler', StandardScaler())])
try:
    categorical_tr = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                               ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
except TypeError:
    categorical_tr = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                               ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

preprocessor = ColumnTransformer([('num', numeric_tr, num_features),
                                  ('cat', categorical_tr, cat_features)],
                                 remainder='passthrough')

emu = GradientBoostingClassifier(n_estimators=200, random_state=42)
emu_pipe = Pipeline([('preprocessor', preprocessor), ('classifier', emu)])
emu_pipe.fit(X_train, y_felix_train)

# 6) Contrôles
fid = accuracy_score(y_felix_test, emu_pipe.predict(X_test))      # fidélité à Félix
auc_baseline = roc_auc_score(y_test, emu_pipe.predict_proba(X_test)[:,1])  # perf vs vérité terrain
print(f"Fidélité à Félix (test): {fid:.3f} | Baseline AUC (vs y): {auc_baseline:.3f}")

# 7) Construire les GROUPES (robuste)
preproc_fitted = emu_pipe.named_steps['preprocessor']
clf = emu_pipe.named_steps['classifier']
feat_names = np.array(preproc_fitted.get_feature_names_out(), dtype=str)

num_cols = list(preproc_fitted.transformers_[0][2]) if len(preproc_fitted.transformers_)>0 else []
cat_cols = list(preproc_fitted.transformers_[1][2]) if len(preproc_fitted.transformers_)>1 else []

group_indices = OrderedDict()
# numériques: 'num__<raw>'
for raw in num_cols:
    hit = np.where(feat_names == f"num__{raw}")[0]
    if hit.size: group_indices[raw] = hit.tolist()

# catégorielles: utiliser categories_ de l'OHE
if len(cat_cols)>0:
    ohe = preproc_fitted.named_transformers_['cat'].named_steps['onehot']
    for raw, cats in zip(cat_cols, ohe.categories_):
        names = [f"cat__{raw}_{str(c)}" for c in cats]
        idxs = []
        for nm in names:
            hit = np.where(feat_names == nm)[0]
            if hit.size: idxs.append(int(hit[0]))
        if idxs: group_indices[raw] = idxs

print(f"{len(group_indices)} groupes construits.")

# 8) PI groupée dans l’espace transformé (rapide)
from sklearn.metrics import roc_auc_score
Xt = preproc_fitted.transform(X_test)
base_auc = roc_auc_score(y_test, clf.predict_proba(Xt)[:,1])
N_REPEATS = 12  # 20 pour la version finale
rng = np.random.RandomState(42)

rows = []
for g, idxs in group_indices.items():
    drops = []
    for _ in range(N_REPEATS):
        Xp = Xt.copy()
        perm = rng.permutation(Xt.shape[0])
        Xp[:, idxs] = Xt[perm][:, idxs]
        score = roc_auc_score(y_test, clf.predict_proba(Xp)[:,1])
        drops.append(base_auc - score)
    rows.append((g, float(np.mean(drops)), float(np.std(drops))))

pi_group_df = pd.DataFrame(rows, columns=["group","pi_mean","pi_std"]).sort_values("pi_mean", ascending=False).reset_index(drop=True)
pi_group_df["ci95_low"]  = pi_group_df["pi_mean"] - 1.96 * pi_group_df["pi_std"]/np.sqrt(N_REPEATS)
pi_group_df["ci95_high"] = pi_group_df["pi_mean"] + 1.96 * pi_group_df["pi_std"]/np.sqrt(N_REPEATS)

# 9) Export + figure
os.makedirs("outputs", exist_ok=True); os.makedirs("graphs", exist_ok=True)
stamp = datetime.now().strftime("%m%d_%H%M")
csv_path = f"outputs/permutation_importance_GROUP_auc_clean_{stamp}.csv"
pi_group_df.to_csv(csv_path, index=False)
print("CSV:", csv_path)

topk = min(15, len(pi_group_df))
sub = pi_group_df.head(topk).iloc[::-1]
plt.figure(figsize=(9, max(6, topk*0.35)))
plt.barh(sub["group"], sub["pi_mean"], xerr=sub["pi_std"])
plt.xlabel("Drop in AUC when permuted (grouped)")
plt.title(f"Permutation Importance — Grouped — top {topk}")
plt.tight_layout()
fig_path = f"graphs/pi_group_bar_clean_top{topk}_{stamp}.png"
plt.savefig(fig_path, dpi=150); plt.close()
print("Figure:", fig_path)


# In[ ]:





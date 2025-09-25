import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# Use LightGBM with a custom objective for differentiable rounding around cutoff
import lightgbm as lgb

# Plotting, saving models/metadata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import json
import os
from datetime import datetime


# Resolve important directories relative to this file
CURRENT_DIR = os.path.dirname(__file__)
GRAPHS_DIR = os.path.join(CURRENT_DIR, 'graphs')
MODEL_DIR = os.path.join(CURRENT_DIR, 'model')
DATA_DIR = os.path.normpath(os.path.join(CURRENT_DIR, '..', 'data'))
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Load the dataset from CSV (preferred for repo)
data_path = os.path.join(DATA_DIR, 'dataproject2025.csv')
try:
    df = pd.read_csv(data_path)
    print("Dataset loaded successfully.")
    print("Shape of the dataset:", df.shape)
except FileNotFoundError:
    raise FileNotFoundError(f"Error: 'dataproject2025.csv' not found at {data_path}.")

# Identify columns to exclude from features
columns_to_exclude = [c for c in ['target', 'Predictions', 'Predicted probabilities', 'Unnamed: 0'] if c in df.columns]
# Drop sensitive attribute(s)
columns_to_exclude += [c for c in df.columns if 'Pct_afro_american' in c]

# The 'target' column is our dependent variable (y)
y = df['target']

# The rest of the columns are potential features (X)
X = df.drop(columns=columns_to_exclude, errors='ignore')

# Preserve the identifier column for later export if present
id_series = df['Unnamed: 0'] if 'Unnamed: 0' in df.columns else pd.Series(np.arange(len(df)), name='Unnamed: 0')

print("Features (X) shape:", X.shape)
print("Target (y) shape:", y.shape)

# Automatically identify numerical and categorical columns
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

print(f"Found {len(numerical_features)} numerical features.")
print(f"Found {len(categorical_features)} categorical features.")

# Create a preprocessing pipeline for numerical features:
# 1. Impute missing values with the median
# 2. Scale features to a standard range
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Create a preprocessing pipeline for categorical features:
# 1. Impute missing values with a constant string 'missing'
# 2. One-hot encode the categories
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False is often helpful for HistGradientBoostingClassifier
])

# Combine preprocessing steps into a single ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

### Cell 6: Define the Full Model Pipeline

# Define the model using LightGBM with explicit l2 (MSE) objective
model = lgb.LGBMRegressor(objective='l2', n_estimators=200, random_state=42)

# Create the full pipeline by chaining the preprocessor and the model
full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', model)])

print("--- Full Model Pipeline ---")
print(full_pipeline)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training the model...")
# The pipeline handles imputation, scaling, and encoding automatically
train_weights = np.where(y_train.values == 1, 2.0, 1.0)
full_pipeline.fit(X_train, y_train, classifier__sample_weight=train_weights)
print("Model training complete.")

print("--- Evaluating Model Performance on Test Data ---")

# Get regression scores and apply cutoff for classification
cutoff = 0.5
y_score = full_pipeline.predict(X_test)
y_pred = (y_score >= cutoff).astype(int)
# Clip scores to [0,1] for interpretability
y_pred_proba = np.clip(y_score, 0.0, 1.0)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC Score: {auc:.4f}")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# --- Compute accuracy over boosting iterations (train and test) ---
print("Computing per-iteration metrics (LightGBM)...")

# Transform datasets using the fitted preprocessor
preprocessor_fitted = full_pipeline.named_steps['preprocessor']
regressor_fitted = full_pipeline.named_steps['classifier']
X_train_processed = preprocessor_fitted.transform(X_train)
X_test_processed = preprocessor_fitted.transform(X_test)

train_accuracies = []
test_accuracies = []
train_mse_rounded = []
test_mse_rounded = []
train_obj_loss = []
test_obj_loss = []
num_iters = getattr(regressor_fitted, 'n_estimators', 0) or 0
if num_iters:
    for i in range(1, num_iters + 1):
        score_train = regressor_fitted.predict(X_train_processed, num_iteration=i)
        score_test = regressor_fitted.predict(X_test_processed, num_iteration=i)
        y_pred_train_iter = (score_train >= cutoff).astype(int)
        y_pred_test_iter = (score_test >= cutoff).astype(int)
        train_accuracies.append(accuracy_score(y_train, y_pred_train_iter))
        test_accuracies.append(accuracy_score(y_test, y_pred_test_iter))
        # Rounded-label MSE (equivalent to misclassification rate)
        train_mse_rounded.append(float(np.mean((y_train.values - y_pred_train_iter) ** 2)))
        test_mse_rounded.append(float(np.mean((y_test.values - y_pred_test_iter) ** 2)))
        # Objective loss used during training: MSE on raw regression predictions (weighted on both splits)
        if 'train_weights' in locals():
            w_train = train_weights
            train_obj = float(np.sum(w_train * (y_train.values - score_train) ** 2) / np.sum(w_train))
        else:
            train_obj = float(np.mean((y_train.values - score_train) ** 2))
        w_test = np.where(y_test.values == 1, 2.0, 1.0)
        test_obj = float(np.sum(w_test * (y_test.values - score_test) ** 2) / np.sum(w_test))
        train_obj_loss.append(train_obj)
        test_obj_loss.append(test_obj)
else:
    print("Warning: Unable to determine number of iterations for per-iteration accuracy plot.")

# Plot and save accuracy-over-iterations graph
timestamp = datetime.now().strftime('%m%d_%H%M')
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Gradient Boosting Accuracy Over Iterations')
plt.legend()
plt.grid(True, alpha=0.3)
graph_path = os.path.join(GRAPHS_DIR, f'accuracy_progress_{timestamp}.png')
plt.tight_layout()
plt.savefig(graph_path, dpi=150)
plt.close()
print(f"Saved accuracy-over-iterations graph to: {graph_path}")

# Dual-panel figure: Objective loss (left) and Rounded-label MSE (right)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
iters = range(1, len(train_accuracies) + 1)

axes[0].plot(iters, train_obj_loss, label='Train Objective (MSE raw)')
axes[0].plot(iters, test_obj_loss, label='Test Objective (MSE raw)')
axes[0].set_xlabel('Number of Estimators')
axes[0].set_ylabel('Objective (MSE on raw predictions)')
axes[0].set_title('Objective loss (LightGBM default: MSE on raw)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(iters, train_mse_rounded, label='Train Rounded MSE')
axes[1].plot(iters, test_mse_rounded, label='Test Rounded MSE')
axes[1].set_xlabel('Number of Estimators')
axes[1].set_ylabel('MSE (rounded labels)')
axes[1].set_title(f'Rounded-label MSE (cutoff={cutoff})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

dual_graph_path = os.path.join(GRAPHS_DIR, f'training_progress_dual_{timestamp}.png')
plt.tight_layout()
plt.savefig(dual_graph_path, dpi=150)
plt.close()
print(f"Saved dual training progress graph to: {dual_graph_path}")

# Persist the trained pipeline and metadata
model_path = os.path.join(MODEL_DIR, f'grad_boost_model_{timestamp}.joblib')
joblib.dump(full_pipeline, model_path)
metadata = {
    'model_type': 'LGBMRegressor_l2',
    'timestamp': timestamp,
    'train_size': int(len(y_train)),
    'test_size': int(len(y_test)),
    'n_features': int(X.shape[1]),
    'n_categorical_features': int(len([c for c in X.columns if X[c].dtype == 'object'])),
    'n_numerical_features': int(len([c for c in X.columns if np.issubdtype(X[c].dtype, np.number)])),
    'test_accuracy': float(accuracy),
    'test_auc': float(auc),
    'graph_path': graph_path,
    'model_path': model_path
}
metadata_path = os.path.join(MODEL_DIR, f'grad_boost_model_{timestamp}_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Saved model to: {model_path}")
print(f"Saved metadata to: {metadata_path}")

# Export required columns to data/ for later use
print("Exporting Unnamed: 0, target, grad_boost_predict to data/ ...")
all_scores = full_pipeline.predict(X)
all_predictions = (all_scores >= cutoff).astype(int)
export_df = pd.DataFrame({
    'Unnamed: 0': id_series.values,
    'target': df['target'].values,
    'grad_boost_predict': all_predictions
})
preds_csv_path = os.path.join(DATA_DIR, f'dataproject2025_grad_boost_predictions_{timestamp}.csv')
export_df.to_csv(preds_csv_path, index=False)
print(f"Saved predictions CSV to: {preds_csv_path}")

# Bar chart: percentage of predicted 0s and 1s across cutoffs
cutoffs = [0.22, 0.26, 0.3, 0.34, 0.38, 0.42, 0.46, 0.5]
perc_ones = []
perc_zeros = []
for c in cutoffs:
    preds_c = (all_scores >= c).astype(int)
    perc_ones.append(float((preds_c == 1).mean()))
    perc_zeros.append(float((preds_c == 0).mean()))

plt.figure(figsize=(7, 5))
x = np.arange(len(cutoffs))
width = 0.38
plt.bar(x - width/2, perc_zeros, width=width, label='Predicted 0')
plt.bar(x + width/2, perc_ones, width=width, label='Predicted 1')
plt.xticks(x, [str(c) for c in cutoffs])
plt.ylim(0, 1)
plt.ylabel('Percentage of predictions')
plt.xlabel('Cutoff value')
plt.title('Predicted class distribution across cutoffs')
plt.legend()
cutoff_graph_path = os.path.join(GRAPHS_DIR, f'predicted_distribution_by_cutoff_{timestamp}.png')
plt.tight_layout()
plt.savefig(cutoff_graph_path, dpi=150)
plt.close()
print(f"Saved cutoff distribution graph to: {cutoff_graph_path}")

# Histogram of raw regression scores with out-of-range highlight
out_low = int((all_scores < 0).sum())
out_high = int((all_scores > 1).sum())
total_scores = int(len(all_scores))

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(all_scores, bins=60, color='steelblue', alpha=0.85)
ax.axvline(0, color='crimson', linestyle='--', linewidth=1.5, label='0/1 bounds')
ax.axvline(1, color='crimson', linestyle='--', linewidth=1.5)
ax.set_xlabel('Raw regression score')
ax.set_ylabel('Count')
ax.set_title('Distribution of regression scores (raw)')
ax.legend()

# Annotate out-of-range counts in axes coordinates
annot = (
    f"< 0: {out_low} ({(out_low/total_scores*100 if total_scores else 0):.2f}%)\n"
    f"> 1: {out_high} ({(out_high/total_scores*100 if total_scores else 0):.2f}%)"
)
ax.text(0.02, 0.98, annot, transform=ax.transAxes, va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

score_hist_path = os.path.join(GRAPHS_DIR, f'regression_score_distribution_{timestamp}.png')
plt.tight_layout()
plt.savefig(score_hist_path, dpi=150)
plt.close()
print(f"Saved regression score distribution histogram to: {score_hist_path}")
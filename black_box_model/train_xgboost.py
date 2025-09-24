import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# Use classic GradientBoostingClassifier to access staged predictions for accuracy-over-iterations
from sklearn.ensemble import GradientBoostingClassifier

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

# Define the model using scikit-learn's GradientBoostingClassifier (supports staged predictions)
model = GradientBoostingClassifier(n_estimators=200, random_state=42)

# Create the full pipeline by chaining the preprocessor and the model
full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', model)])

print("--- Full Model Pipeline ---")
print(full_pipeline)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training the model...")
# The pipeline handles imputation, scaling, and encoding automatically
full_pipeline.fit(X_train, y_train)
print("Model training complete.")

print("--- Evaluating Model Performance on Test Data ---")

# Get predictions
y_pred = full_pipeline.predict(X_test)
y_pred_proba = full_pipeline.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC Score: {auc:.4f}")
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# --- Compute accuracy over boosting iterations (train and test) ---
print("Computing staged accuracies over boosting iterations...")

# Transform datasets using the fitted preprocessor
preprocessor_fitted = full_pipeline.named_steps['preprocessor']
classifier_fitted = full_pipeline.named_steps['classifier']
X_train_processed = preprocessor_fitted.transform(X_train)
X_test_processed = preprocessor_fitted.transform(X_test)

train_accuracies = []
test_accuracies = []
for y_pred_train, y_pred_test in zip(
        classifier_fitted.staged_predict(X_train_processed),
        classifier_fitted.staged_predict(X_test_processed)):
    train_accuracies.append(accuracy_score(y_train, y_pred_train))
    test_accuracies.append(accuracy_score(y_test, y_pred_test))

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

# Persist the trained pipeline and metadata
model_path = os.path.join(MODEL_DIR, f'grad_boost_model_{timestamp}.joblib')
joblib.dump(full_pipeline, model_path)
metadata = {
    'model_type': 'GradientBoostingClassifier',
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
all_predictions = full_pipeline.predict(X)
export_df = pd.DataFrame({
    'Unnamed: 0': id_series.values,
    'target': df['target'].values,
    'grad_boost_predict': all_predictions
})
preds_csv_path = os.path.join(DATA_DIR, f'dataproject2025_grad_boost_predictions_{timestamp}.csv')
export_df.to_csv(preds_csv_path, index=False)
print(f"Saved predictions CSV to: {preds_csv_path}")
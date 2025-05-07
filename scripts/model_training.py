# train_model.py

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# Set base and output paths using pathlib
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR.parent / 'processed_data'
MODELS_DIR = BASE_DIR.parent / 'models'
VISUALS_DIR = BASE_DIR.parent / 'visualizations'

# Ensure output directories exist
MODELS_DIR.mkdir(exist_ok=True)
VISUALS_DIR.mkdir(exist_ok=True)

# Load final processed data
X = pd.read_csv(PROCESSED_DIR / 'X_processed.csv')
y = pd.read_csv(PROCESSED_DIR / 'y_processed.csv').values.ravel()

print(f"\n✅ Data loaded: X={X.shape}, y={y.shape}")

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for deployment use
joblib.dump(scaler, MODELS_DIR / 'scaler.joblib')     
print("✅ Scaler saved successfully.")

# Define Logistic Regression + GridSearchCV
model = LogisticRegression()
param_grid = {
    'penalty': ['l1', 'l2'],    
    'solver': ['liblinear'],
    'C': [0.01, 0.1, 1, 10, 100],
    'class_weight': [None, 'balanced'],
    'max_iter': [1000]
}

grid_search = GridSearchCV(model, param_grid, scoring='f1', cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_scaled, y)

print("\n✅ Grid search completed.")
print("Best Parameters:", grid_search.best_params_)
print("📈 Best CV F1 Score (from GridSearch):", round(grid_search.best_score_, 4))

# Get best model
best_model = grid_search.best_estimator_

# Step 1: Evaluate with cross_val_score
cv_f1_scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='f1')
print("\n📊 Cross-Validated F1 Scores:", np.round(cv_f1_scores, 4))
print("📈 Mean F1:", round(cv_f1_scores.mean(), 4), "| Std Dev:", round(cv_f1_scores.std(), 4))

# Step 2: Predict with cross_val_predict for confusion matrix
y_cv_pred = cross_val_predict(best_model, X_scaled, y, cv=5)
print("\n📊 Classification Report (Cross-Validated):")
print(classification_report(y, y_cv_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y, y_cv_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Eligible", "Eligible"], yticklabels=["Not Eligible", "Eligible"])
plt.title("Cross-Validated Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(VISUALS_DIR / 'confusion_matrix.png')
plt.close()
print("✅ Cross-validated confusion matrix saved to 'visualizations/confusion_matrix.png'")

# Save best model
joblib.dump(best_model, MODELS_DIR / 'loan_eligibility_model.joblib')
print("\n✅ Best model saved as 'loan_eligibility_model.joblib'")

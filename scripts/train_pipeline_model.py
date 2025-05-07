from pathlib import Path
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from scripts.preprocessing_pipeline import LoanPreprocessor


# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / 'data' / 'loan-train.csv'
MODEL_PATH = BASE_DIR.parent / 'models' / 'pipeline.joblib'
MODEL_PATH.parent.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Define pipeline
pipeline = Pipeline([
    ('preprocessing', LoanPreprocessor()),
    ('scaling', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

# Grid search (optional)
param_grid = {
    'model__C': [0.01, 0.1, 1, 10],
    'model__penalty': ['l1', 'l2'],
    'model__solver': ['liblinear'],
    'model__class_weight': [None, 'balanced']
}

grid = GridSearchCV(pipeline, param_grid, scoring='f1', cv=5, n_jobs=-1, verbose=1)
grid.fit(X, y)

# Save best pipeline
joblib.dump(grid.best_estimator_, MODEL_PATH)
print("✅ Pipeline saved to:", MODEL_PATH)
print("📈 Best Params:", grid.best_params_)


from pathlib import Path
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Define base directory using pathlib
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / 'data' / 'loan-train.csv'
PROCESSED_DIR = BASE_DIR.parent / 'processed_data'


def load_raw_data(filepath):
    """Load the original loan eligibility dataset."""
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    """Perform data cleaning: drop ID, fill missing values."""
    df.drop('Loan_ID', axis=1, inplace=True)
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna('No', inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    return df

def encode_features(df):
    """Encode categorical features appropriately."""
    binary_features = ['Gender', 'Married', 'Education', 'Self_Employed']
    mapping = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0, 'Graduate': 1, 'Not Graduate': 0}
    for feature in binary_features:
        df[feature] = df[feature].map(mapping)
    multi_class_features = ['Dependents', 'Property_Area']
    df = pd.get_dummies(df, columns=multi_class_features, drop_first=True)
    return df

def add_engineered_features(df):
    """Add domain-specific features like Total_Income and LoanToIncomeRatio."""
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['LoanToIncomeRatio'] = df['LoanAmount'] / (df['Total_Income'] + 1e-5)
    return df

def transform_features(df):
    """Apply log transformation to skewed numerical features."""
    skewed_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Total_Income', 'LoanToIncomeRatio']
    for feature in skewed_features:
        df[feature] = np.log1p(df[feature])
    return df

def split_features_target(df):
    """Split DataFrame into features (X) and target (y)."""
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status'].map({'Y': 1, 'N': 0})
    return X, y

def save_processed_data(X, y):
    """Save processed X and y to CSV files."""
    PROCESSED_DIR.mkdir(exist_ok=True)
    X.to_csv(PROCESSED_DIR / 'X_processed.csv', index=False)
    y.to_csv(PROCESSED_DIR / 'y_processed.csv', index=False)

def main():
    df = load_raw_data(DATA_PATH)
    df = clean_data(df)
    df = encode_features(df)
    df = add_engineered_features(df)
    df = transform_features(df)
    X, y = split_features_target(df)
    save_processed_data(X, y)
    print("\n✅ Preprocessing completed successfully! Processed files saved.")
    print("✅ Columns after feature engineering:", df.columns.tolist())

if __name__ == '__main__':
    main()

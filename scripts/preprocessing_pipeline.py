from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class LoanPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_ = None

    def fit(self, X, y=None):
        # Run a full transform on the training set to capture all columns:
        df = self._transform(pd.DataFrame(X).copy())
        self.feature_names_ = df.columns.tolist()
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        df = self._transform(df)
        # Reindex to training columns, filling new rows with 0
        df = df.reindex(columns=self.feature_names_, fill_value=0)
        return df

    def _transform(self, df):
        """All your existing steps, without reindexing."""
        # 1) Drop Loan_ID
        if 'Loan_ID' in df.columns:
            df.drop('Loan_ID', axis=1, inplace=True)

        # 2) Fill missing
        df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
        df['Married'].fillna(df['Married'].mode()[0], inplace=True)
        df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
        df['Self_Employed'].fillna('No', inplace=True)
        df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
        df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
        df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

        # 3) Binary encode
        mapping = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0, 
                   'Graduate': 1, 'Not Graduate': 0}
        for col in ['Gender', 'Married', 'Education', 'Self_Employed']:
            df[col] = df[col].map(mapping)

        # 4) One-hot encode multi-class
        df = pd.get_dummies(df, columns=['Dependents', 'Property_Area'], drop_first=True)

        # 5) Engineered features
        df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        df['LoanToIncomeRatio'] = df['LoanAmount'] / (df['Total_Income'] + 1e-5)

        # 6) Log transform skewed
        for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                    'Total_Income', 'LoanToIncomeRatio']:
            df[col] = np.log1p(df[col])

        return df

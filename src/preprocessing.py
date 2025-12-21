import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(train_path, test_path):
    """Loads train and test datasets for credit risk prediction."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def get_preprocessor():
    """Returns a ColumnTransformer for Kaggle loan approval data preprocessing."""
    # Numerical features from Kaggle dataset
    numerical_cols = [
        'person_age',                    # Age of the person
        'person_income',                 # Annual income
        'person_emp_length',             # Employment length in years
        'loan_amnt',                     # Loan amount
        'loan_int_rate',                 # Loan interest rate
        'loan_percent_income',           # Loan amount as percentage of income
        'cb_person_cred_hist_length'     # Credit history length in years
    ]
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical features from Kaggle dataset
    categorical_cols = [
        'person_home_ownership',         # Home ownership status (RENT, OWN, MORTGAGE)
        'loan_intent',                   # Loan purpose (EDUCATION, MEDICAL, PERSONAL, etc.)
        'loan_grade',                    # Loan grade (A, B, C, D, E, F)
        'cb_person_default_on_file'      # Default history (Y/N)
    ]
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    return preprocessor

def preprocess_data(train_df, test_df):
    """Preprocesses Kaggle loan approval data and returns X_train, y_train, X_test."""
    # Drop id and target from training data
    X = train_df.drop(['id', 'loan_status'], axis=1)
    y = train_df['loan_status']  # Already binary (0/1), no encoding needed

    # Drop id from test data
    X_test = test_df.drop(['id'], axis=1)

    # Target is already numeric (0 = no default, 1 = default)
    # No need for LabelEncoder
    return X, y, X_test

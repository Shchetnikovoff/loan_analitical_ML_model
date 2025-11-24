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
    """Returns a ColumnTransformer for credit risk data preprocessing."""
    # Numerical features
    numerical_cols = ['Age', 'Income', 'Credit_Amount', 'Loan_Duration', 'Debt_to_Income', 'Credit_Score', 'Num_Credits', 'Savings_Account_Balance']
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical features
    categorical_cols = ['Gender', 'Employment_Status', 'Education_Level', 'Marital_Status', 'Housing_Type', 'Loan_Purpose']
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
    """Preprocesses credit risk data and returns X_train, y_train, X_test, label_encoder."""
    X = train_df.drop(['ID', 'Credit_Risk'], axis=1)
    y = train_df['Credit_Risk']
    X_test = test_df.drop(['ID'], axis=1)

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, X_test, le

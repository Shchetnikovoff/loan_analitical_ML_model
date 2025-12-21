import pandas as pd
import numpy as np
import random
from datetime import datetime

def generate_synthetic_credit_data(n_samples=1000):
    """Generate synthetic credit risk data"""
    
    np.random.seed(42)
    random.seed(42)
    
    data = {
        'ID': range(1, n_samples + 1),
        'Age': np.random.randint(18, 80, n_samples),
        'Income': np.random.normal(50000, 20000, n_samples),
        'Credit_Amount': np.random.normal(15000, 8000, n_samples),
        'Loan_Duration': np.random.randint(6, 60, n_samples),  # months
        'Debt_to_Income': np.random.uniform(0, 0.8, n_samples),
        'Credit_Score': np.random.randint(300, 850, n_samples),
        'Num_Credits': np.random.randint(0, 10, n_samples),
        'Savings_Account_Balance': np.random.normal(5000, 3000, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Employment_Status': np.random.choice(['Employed', 'Self_Employed', 'Unemployed', 'Retired'], n_samples, p=[0.7, 0.1, 0.1, 0.1]),
        'Education_Level': np.random.choice(['Below_Secondary', 'Secondary', 'Bachelor', 'Master', 'Doctor'], n_samples, p=[0.1, 0.3, 0.3, 0.2, 0.1]),
        'Marital_Status': np.random.choice(['Single', 'Married', 'Divorced', 'Widow'], n_samples, p=[0.3, 0.5, 0.15, 0.05]),
        'Housing_Type': np.random.choice(['Own', 'Rent', 'Mortgage'], n_samples, p=[0.4, 0.4, 0.2]),
        'Loan_Purpose': np.random.choice(['Home', 'Car', 'Education', 'Medical', 'Personal'], n_samples, p=[0.25, 0.25, 0.15, 0.15, 0.2])
    }
    
    # Make Income, Credit_Amount and Savings_Account_Balance non-negative
    data['Income'] = np.abs(data['Income'])
    data['Credit_Amount'] = np.abs(data['Credit_Amount'])
    data['Savings_Account_Balance'] = np.abs(data['Savings_Account_Balance'])
    
    df = pd.DataFrame(data)
    
    # Create Credit_Risk based on financial indicators
    # Higher risk if: high debt-to-income, low credit score, low income relative to loan amount
    risk_score = (
        df['Debt_to_Income'] * 0.4 +
        (850 - df['Credit_Score']) / 850 * 0.3 +
        (df['Credit_Amount'] / df['Income']) * 0.3
    )
    
    # Define risk categories
    conditions = [
        risk_score <= 0.3,
        (risk_score > 0.3) & (risk_score <= 0.6),
        risk_score > 0.6
    ]
    choices = ['Low', 'Medium', 'High']

    # Use pandas cut function to assign risk categories
    df['Credit_Risk'] = pd.cut(risk_score,
                              bins=[-np.inf, 0.3, 0.6, np.inf],
                              labels=['Low', 'Medium', 'High'],
                              include_lowest=True)

    # Convert to string to ensure consistent type
    df['Credit_Risk'] = df['Credit_Risk'].astype(str)
    
    # Make sure all categories are present
    if len(df['Credit_Risk'].unique()) < 3:
        # Force at least a few examples of each category
        low_idx = df.nsmallest(max(5, int(0.1 * n_samples)), 'Debt_to_Income').index
        df.loc[low_idx, 'Credit_Risk'] = 'Low'
        
        high_idx = df.nlargest(max(5, int(0.1 * n_samples)), 'Debt_to_Income').index
        df.loc[high_idx, 'Credit_Risk'] = 'High'
    
    return df

def create_test_data(train_df, n_test_samples=200):
    """Create test data based on train data characteristics"""
    test_data = {
        'ID': range(train_df['ID'].max() + 1, train_df['ID'].max() + n_test_samples + 1),
        'Age': np.random.randint(18, 80, n_test_samples),
        'Income': np.random.normal(50000, 20000, n_test_samples),
        'Credit_Amount': np.random.normal(15000, 8000, n_test_samples),
        'Loan_Duration': np.random.randint(6, 60, n_test_samples),
        'Debt_to_Income': np.random.uniform(0, 0.8, n_test_samples),
        'Credit_Score': np.random.randint(300, 850, n_test_samples),
        'Num_Credits': np.random.randint(0, 10, n_test_samples),
        'Savings_Account_Balance': np.random.normal(5000, 3000, n_test_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_test_samples),
        'Employment_Status': np.random.choice(['Employed', 'Self_Employed', 'Unemployed', 'Retired'], n_test_samples, p=[0.7, 0.1, 0.1, 0.1]),
        'Education_Level': np.random.choice(['Below_Secondary', 'Secondary', 'Bachelor', 'Master', 'Doctor'], n_test_samples, p=[0.1, 0.3, 0.3, 0.2, 0.1]),
        'Marital_Status': np.random.choice(['Single', 'Married', 'Divorced', 'Widow'], n_test_samples, p=[0.3, 0.5, 0.15, 0.05]),
        'Housing_Type': np.random.choice(['Own', 'Rent', 'Mortgage'], n_test_samples, p=[0.4, 0.4, 0.2]),
        'Loan_Purpose': np.random.choice(['Home', 'Car', 'Education', 'Medical', 'Personal'], n_test_samples, p=[0.25, 0.25, 0.15, 0.15, 0.2])
    }
    
    # Make Income, Credit_Amount and Savings_Account_Balance non-negative
    test_data['Income'] = np.abs(test_data['Income'])
    test_data['Credit_Amount'] = np.abs(test_data['Credit_Amount'])
    test_data['Savings_Account_Balance'] = np.abs(test_data['Savings_Account_Balance'])
    
    test_df = pd.DataFrame(test_data)
    
    return test_df

if __name__ == "__main__":
    print("Generating synthetic credit risk data...")
    
    # Generate training and test data
    train_df = generate_synthetic_credit_data(800)  # 800 samples for training
    test_df = create_test_data(train_df, 200)  # 200 samples for testing
    
    # Save to CSV files
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Credit risk distribution in training set:")
    print(train_df['Credit_Risk'].value_counts())
    
    print("\nSynthetic credit risk data generated successfully!")
    print("Files saved as:")
    print("- data/train.csv")
    print("- data/test.csv")
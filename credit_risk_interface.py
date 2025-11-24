import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Add src directory to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

def get_user_input():
    """Get client data from user input"""
    print("Enter client data for credit risk assessment:")

    client_data = {}

    # Numeric inputs
    client_data['Age'] = float(input("Age (years): "))
    client_data['Income'] = float(input("Income ($): "))
    client_data['Credit_Amount'] = float(input("Credit amount ($): "))
    client_data['Loan_Duration'] = int(input("Loan duration (months): "))
    client_data['Debt_to_Income'] = float(input("Debt-to-income ratio (0-1): "))
    client_data['Credit_Score'] = float(input("Credit score (300-850): "))
    client_data['Num_Credits'] = int(input("Number of credits: "))
    client_data['Savings_Account_Balance'] = float(input("Savings account balance ($): "))

    # Categorical inputs
    print("\nSelect gender:")
    print("1. Male")
    print("2. Female")
    gender_choice = input("Enter 1 or 2: ")
    client_data['Gender'] = 'Male' if gender_choice == '1' else 'Female'

    print("\nSelect employment status:")
    print("1. Employed")
    print("2. Unemployed")
    print("3. Self-employed")
    print("4. Retired")
    emp_choices = {'1': 'Employed', '2': 'Unemployed', '3': 'Self_Employed', '4': 'Retired'}
    emp_choice = input("Enter number: ")
    client_data['Employment_Status'] = emp_choices.get(emp_choice, 'Employed')

    print("\nSelect education level:")
    print("1. Below Secondary")
    print("2. Secondary")
    print("3. Bachelor")
    print("4. Master")
    print("5. Doctor")
    edu_choices = {'1': 'Below_Secondary', '2': 'Secondary', '3': 'Bachelor', '4': 'Master', '5': 'Doctor'}
    edu_choice = input("Enter number: ")
    client_data['Education_Level'] = edu_choices.get(edu_choice, 'Secondary')

    print("\nSelect marital status:")
    print("1. Single")
    print("2. Married")
    print("3. Widow")
    print("4. Divorced")
    mar_choices = {'1': 'Single', '2': 'Married', '3': 'Widow', '4': 'Divorced'}
    mar_choice = input("Enter number: ")
    client_data['Marital_Status'] = mar_choices.get(mar_choice, 'Single')

    print("\nSelect housing type:")
    print("1. Own")
    print("2. Rent")
    print("3. Mortgage")
    house_choices = {'1': 'Own', '2': 'Rent', '3': 'Mortgage'}
    house_choice = input("Enter number: ")
    client_data['Housing_Type'] = house_choices.get(house_choice, 'Own')

    print("\nSelect loan purpose:")
    print("1. Home")
    print("2. Car")
    print("3. Education")
    print("4. Medical")
    print("5. Personal")
    loan_choices = {'1': 'Home', '2': 'Car', '3': 'Education', '4': 'Medical', '5': 'Personal'}
    loan_choice = input("Enter number: ")
    client_data['Loan_Purpose'] = loan_choices.get(loan_choice, 'Personal')

    return client_data

def predict_credit_risk(client_data, model, preprocessor, label_encoder):
    """Make prediction for a new client"""
    try:
        # Convert client data to DataFrame with the same column structure as training data
        client_df = pd.DataFrame([client_data])
        
        # Transform the client data using the fitted preprocessor
        client_processed = preprocessor.transform(client_df)
        
        # Predict
        prediction = model.predict(client_processed)[0]
        prediction_proba = model.predict_proba(client_processed)[0]
        
        # Decode prediction
        risk_label = label_encoder.inverse_transform([prediction])[0]
        
        return risk_label, prediction_proba
    except Exception as e:
        print(f"Ошибка при предсказании: {e}")
        return None, None

def main():
    print("="*60)
    print("CREDIT RISK ASSESSMENT SYSTEM")
    print("="*60)

    print("\nLoading data...")
    DATA_DIR = 'data'
    TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')

    if not os.path.exists(TRAIN_PATH):
        print(f"Error: train.csv file not found in {DATA_DIR} folder")
        print("Please ensure that data file is in the data folder")
        return

    try:
        train_df = pd.read_csv(TRAIN_PATH)
        print(f"Data loaded: {train_df.shape[0]} records, {train_df.shape[1]} features")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print("\nPreparing model...")

    # Prepare data for training
    X = train_df.drop(['ID', 'Credit_Risk'], axis=1)
    y = train_df['Credit_Risk']

    # Create preprocessor
    numerical_cols = ['Age', 'Income', 'Credit_Amount', 'Loan_Duration', 'Debt_to_Income', 'Credit_Score', 'Num_Credits', 'Savings_Account_Balance']
    categorical_cols = ['Gender', 'Employment_Status', 'Education_Level', 'Marital_Status', 'Housing_Type', 'Loan_Purpose']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Fit preprocessor
    X_processed = preprocessor.fit_transform(X)

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("\nTraining model...")

    # Train model with GridSearch
    gbc = GradientBoostingClassifier(random_state=42)

    # Reduced parameter grid for faster training
    param_grid = {
        'n_estimators': [50],
        'learning_rate': [0.1],
        'max_depth': [3]
    }

    grid_search = GridSearchCV(
        estimator=gbc,
        param_grid=param_grid,
        cv=2,  # Reduced folds for faster training
        scoring='accuracy',
        verbose=1,
        n_jobs=1
    )

    grid_search.fit(X_processed, y_encoded)
    model = grid_search.best_estimator_

    print(f"Model trained. Best parameters: {grid_search.best_params_}")

    print("\nModel is ready for credit risk assessment!")
    
    while True:
        print("\n" + "="*60)
        client_data = get_user_input()
        
        risk, probabilities = predict_credit_risk(client_data, model, preprocessor, le)
        
        if risk is not None:
            print(f"\nASSESSMENT RESULTS:")
            print(f"Credit Risk: {risk}")

            # Show probabilities for each class
            classes = le.classes_
            print("Probabilities for each class:")
            for i, class_name in enumerate(classes):
                print(f"  {class_name}: {probabilities[i]:.4f}")

            # Provide recommendation based on risk
            print("\nRECOMMENDATION:")
            if risk == 'Low':
                print("  ✅ Credit can be issued - low risk")
            elif risk == 'Medium':
                print("  ⚠️  Credit can be issued with caution - medium risk")
            else:
                print("  ❌ Deny credit - high risk")
        else:
            print("Could not make prediction")

        continue_choice = input("\nDo you want to assess another client? (y/n): ")
        if continue_choice.lower() != 'y':
            break

    print("\nThank you for using the credit risk assessment system!")

if __name__ == "__main__":
    main()
# Loan Approval Prediction - Kaggle Competition

## Overview
This project implements a machine learning system to predict loan approval status based on client financial and personal data for the Kaggle Playground Series S4E10 competition. The system uses Gradient Boosting Classifier with hyperparameter tuning via GridSearchCV.

## Features
- **Machine Learning Model**: Gradient Boosting Classifier with GridSearchCV hyperparameter tuning
- **Interactive Interface**: Command-line interface for credit risk assessment of new clients
- **Comprehensive Pipeline**: Includes data preprocessing, model training, evaluation, and prediction
- **Risk Categories**: Low, Medium, and High risk classification

## Results
- **Validation Accuracy**: 95.17%
- **Validation ROC-AUC**: 95.76%
- **Best Parameters**:
  - learning_rate: 0.1
  - max_depth: 5
  - n_estimators: 200

## Setup
1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Kaggle data**:
    - Download competition data from: https://www.kaggle.com/competitions/playground-series-s4e10/data
    - Place `train.csv` and `test.csv` in `data/playground-series-s4e10/`

## Usage

### 1. Interactive Credit Risk Assessment
Run the interactive client interface:
```bash
python credit_risk_interface.py
```
This will allow you to input client data and get credit risk predictions.

### 2. Training and Submission
To train the model and generate a submission:
```bash
cd src
python train.py
```

### 3. Generate Jupyter Notebook
To generate the analysis notebook:
```bash
python create_notebook.py
```

### 4. Validate Full System
To run a comprehensive test of the system:
```bash
python test_full_system.py
```

## Project Structure
-   `data/`: Contains train.csv and test.csv with credit risk data
-   `src/`: Source code for preprocessing and modeling
    -   `model.py`: Machine learning model implementation
    -   `preprocessing.py`: Data preprocessing functions
    -   `train.py`: Main training script
-   `notebooks/`: Jupyter notebooks for EDA and credit risk analysis
-   `submission.csv`: Generated submission file
-   `create_notebook.py`: Script to generate the Jupyter notebook
-   `credit_risk_interface.py`: Interactive client interface
-   `generate_data.py`: Script to create synthetic credit risk data
-   `test_full_system.py`: Comprehensive system testing script

## Case 9 Requirements Implementation
✅ **Algorithm**: Gradient Boosting Classifier
✅ **Hyperparameter Tuning**: GridSearchCV with parameter grid
✅ **Data Preprocessing**: Numerical and categorical feature handling
✅ **Model Evaluation**: Accuracy metrics and classification report
✅ **Interactive Interface**: For client credit risk assessment

## Credit Risk Categories
- **Low**: Very low risk, credit can be issued
- **Medium**: Moderate risk, credit can be issued with caution
- **High**: High risk, credit should be denied

## Input Features (Kaggle Dataset)

**Numerical (7 features)**:
- person_age: Age of the person
- person_income: Annual income
- person_emp_length: Employment length in years
- loan_amnt: Loan amount
- loan_int_rate: Loan interest rate
- loan_percent_income: Loan amount as percentage of income
- cb_person_cred_hist_length: Credit history length in years

**Categorical (4 features)**:
- person_home_ownership: Home ownership status (RENT, OWN, MORTGAGE)
- loan_intent: Loan purpose (EDUCATION, MEDICAL, PERSONAL, VENTURE, DEBTCONSOLIDATION, HOMEIMPROVEMENT)
- loan_grade: Loan grade (A, B, C, D, E, F)
- cb_person_default_on_file: Default history (Y/N)

**Target Variable**:
- loan_status: 0 = no default, 1 = default

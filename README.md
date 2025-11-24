# Credit Risk Assessment System - Case 9

## Overview
This project implements a machine learning system to predict credit risk based on client financial and personal data. The system uses Gradient Boosting Classifier with hyperparameter tuning via GridSearchCV as required by Case 9.

## Features
- **Machine Learning Model**: Gradient Boosting Classifier with GridSearchCV hyperparameter tuning
- **Interactive Interface**: Command-line interface for credit risk assessment of new clients
- **Comprehensive Pipeline**: Includes data preprocessing, model training, evaluation, and prediction
- **Risk Categories**: Low, Medium, and High risk classification

## Setup
1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Generate synthetic data** (if not already done):
    ```bash
    python generate_data.py
    ```

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

## Input Features
**Numerical**:
- Age, Income, Credit Amount, Loan Duration, Debt-to-Income ratio, Credit Score, Number of Credits, Savings Account Balance

**Categorical**:
- Gender, Employment Status, Education Level, Marital Status, Housing Type, Loan Purpose

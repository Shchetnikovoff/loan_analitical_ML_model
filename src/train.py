import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from preprocessing import load_data, preprocess_data, get_preprocessor
from model import train_model, evaluate_model

def main():
    # Paths for Kaggle dataset
    DATA_DIR = '../data/playground-series-s4e10'
    TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
    TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
    SUBMISSION_PATH = '../submission.csv'
    MODEL_PATH = '../best_model.pkl'

    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        print(f"Error: Data files not found in {DATA_DIR}. Please download Kaggle competition data.")
        return

    # 1. Load Data
    print("Loading Kaggle loan approval data...")
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # 2. Preprocessing
    print("Preprocessing loan approval data...")
    X, y, X_test_raw = preprocess_data(train_df, test_df)

    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Transform features
    preprocessor = get_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test_raw)

    # 3. Modeling - Gradient Boosting with GridSearchCV
    print("Training loan approval model (Gradient Boosting with GridSearchCV)...")
    best_model, best_params = train_model(X_train_processed, y_train)
    print(f"Best Parameters: {best_params}")

    # 4. Evaluation
    print("\nEvaluating loan approval model...")
    evaluate_model(best_model, X_val_processed, y_val)

    # Save the best model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\nBest model saved to {MODEL_PATH}")

    # 5. Submission for Kaggle
    print("\nGenerating Kaggle submission...")
    predictions = best_model.predict(X_test_processed)

    submission = pd.DataFrame({
        'id': test_df['id'],
        'loan_status': predictions
    })
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(f"Submission shape: {submission.shape}")
    print(f"Prediction distribution: {submission['loan_status'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()

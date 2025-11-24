import pandas as pd
import os
from sklearn.model_selection import train_test_split
from preprocessing import load_data, preprocess_data, get_preprocessor
from model import train_model, evaluate_model

def main():
    # Paths
    DATA_DIR = '../data'
    TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
    TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
    SUBMISSION_PATH = '../submission.csv'

    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        print(f"Error: Data files not found in {DATA_DIR}. Please download them from Kaggle.")
        return

    # 1. Load Data
    print("Loading data...")
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

    # 2. Preprocessing
    print("Preprocessing data...")
    X, y, X_test_raw, label_encoder = preprocess_data(train_df, test_df)
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Transform features
    preprocessor = get_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test_raw)

    # 3. Modeling (Case 9)
    print("Training model (Gradient Boosting with GridSearchCV)...")
    best_model, best_params = train_model(X_train_processed, y_train)
    print(f"Best Parameters: {best_params}")

    # 4. Evaluation
    print("Evaluating model...")
    evaluate_model(best_model, X_val_processed, y_val, label_encoder)

    # 5. Submission
    print("Generating submission...")
    predictions = best_model.predict(X_test_processed)
    predictions_decoded = label_encoder.inverse_transform(predictions)
    
    submission = pd.DataFrame({'id': test_df['id'], 'NObeyesdad': predictions_decoded})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")

if __name__ == "__main__":
    main()

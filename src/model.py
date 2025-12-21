from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

def train_model(X_train, y_train):
    """Trains Gradient Boosting Classifier with GridSearchCV for credit risk prediction."""
    gbc = GradientBoostingClassifier(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }

    # Use roc_auc as scoring metric for Kaggle competition
    grid_search = GridSearchCV(estimator=gbc, param_grid=param_grid, cv=3, scoring='roc_auc', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model, X_val, y_val):
    """Evaluates the loan approval model and prints metrics."""
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation ROC-AUC: {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['No Default (0)', 'Default (1)']))

    return acc, auc

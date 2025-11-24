from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model(X_train, y_train):
    """Trains Gradient Boosting Classifier with GridSearchCV for credit risk prediction."""
    gbc = GradientBoostingClassifier(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }

    grid_search = GridSearchCV(estimator=gbc, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model, X_val, y_val, label_encoder):
    """Evaluates the credit risk model and prints metrics."""
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

    return acc

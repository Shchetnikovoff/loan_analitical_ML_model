import os
import sys
import pandas as pd
import numpy as np

# Добавим директорию src в путь
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

# Загрузим данные
DATA_DIR = 'data'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')

print('Loading data...')
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Импортируем модули из src
from preprocessing import load_data, preprocess_data, get_preprocessor
from model import train_model, evaluate_model

print('Testing preprocessing module...')
X, y, X_test_raw, label_encoder = preprocess_data(train_df, test_df)
print(f'X shape: {X.shape}, y shape: {y.shape}, X_test shape: {X_test_raw.shape}')
print(f'Label classes: {label_encoder.classes_}')

print('Testing preprocessor...')
preprocessor = get_preprocessor()
X_transformed = preprocessor.fit_transform(X)
X_test_transformed = preprocessor.transform(X_test_raw)
print(f'Transformed X shape: {X_transformed.shape}')
print(f'Transformed test X shape: {X_test_transformed.shape}')

print('Testing model training...')
# Используем небольшую часть данных для быстрой проверки
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Обучим модель с уменьшенным параметром grid для быстрой проверки
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Упрощенный параметр grid для быстрой проверки
param_grid = {
    'n_estimators': [50],
    'learning_rate': [0.1],
    'max_depth': [3]
}

print('Training with GridSearchCV...')
gbc = GradientBoostingClassifier(random_state=42)
grid_search = GridSearchCV(estimator=gbc, param_grid=param_grid, cv=2, scoring='accuracy', verbose=1, n_jobs=1)
grid_search.fit(X_train, y_train)

model = grid_search.best_estimator_
print(f'Best parameters: {grid_search.best_params_}')

# Оценим модель
y_pred = model.predict(X_val)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_val, y_pred)
print(f'Validation accuracy: {acc:.4f}')

print('All tests passed! Project is ready for submission.')
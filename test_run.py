import os
import sys
import pandas as pd

# Добавим директорию src в путь
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

# Определим пути как в оригинальном скрипте
DATA_DIR = 'data'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')

print('Checking if files exist:')
print(f'train.csv exists: {os.path.exists(TRAIN_PATH)}')
print(f'test.csv exists: {os.path.exists(TEST_PATH)}')

# Если файлы существуют, попробуем их загрузить
if os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH):
    print('Loading data...')
    try:
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
        print(f'Training dataset shape: {train_df.shape}')
        print(f'Test dataset shape: {test_df.shape}')
        print('Loading successful!')

        # Проверим первые несколько строк
        print('First rows of training dataset:')
        print(train_df.head())

        print('First rows of test dataset:')
        print(test_df.head())

        # Проверим, что все необходимые колонки присутствуют
        required_cols_train = ['id', 'NObeyesdad', 'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE',
                               'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
        missing_cols_train = [col for col in required_cols_train if col not in train_df.columns]

        if missing_cols_train:
            print(f'Missing columns in training dataset: {missing_cols_train}')
        else:
            print('All required columns present in training dataset')

        required_cols_test = ['id', 'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE',
                              'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
        missing_cols_test = [col for col in required_cols_test if col not in test_df.columns]

        if missing_cols_test:
            print(f'Missing columns in test dataset: {missing_cols_test}')
        else:
            print('All required columns present in test dataset')

    except Exception as e:
        print(f'Error loading data: {e}')
else:
    print('Files not found')
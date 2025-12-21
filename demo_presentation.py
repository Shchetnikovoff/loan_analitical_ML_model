"""
Интерактивная демонстрация решения Kaggle Competition
Loan Approval Prediction - Playground Series S4E10

Запуск: python demo_presentation.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import time
from sklearn.model_selection import train_test_split
import sys

# Добавляем src в path
sys.path.insert(0, 'src')
from preprocessing import load_data, preprocess_data, get_preprocessor
from model import train_model, evaluate_model


def print_header(text, char="="):
    """Печать красивого заголовка"""
    width = 80
    print("\n" + char * width)
    print(text.center(width))
    print(char * width + "\n")


def print_section(text):
    """Печать секции"""
    print("\n" + "─" * 80)
    print(f">>> {text}")
    print("─" * 80)


def wait_for_enter(message="Нажмите Enter для продолжения..."):
    """Пауза для демонстрации"""
    input(f"\n{message}")


def animated_print(text, delay=0.03):
    """Печать с анимацией"""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()


def show_progress_bar(text, duration=2):
    """Показать прогресс-бар"""
    print(f"\n{text}")
    bar_length = 50
    for i in range(bar_length + 1):
        percent = (i / bar_length) * 100
        bar = "█" * i + "░" * (bar_length - i)
        print(f"\r[{bar}] {percent:.0f}%", end='', flush=True)
        time.sleep(duration / bar_length)
    print("\n")


def main():
    # Вступление
    print_header("ДЕМОНСТРАЦИЯ РЕШЕНИЯ KAGGLE COMPETITION", "=")
    print_header("Loan Approval Prediction - Playground Series S4E10", " ")

    animated_print("Автор: Shchetnikovoff")
    animated_print("GitHub: https://github.com/Shchetnikovoff/loan_analitical_ML_model.git")

    wait_for_enter("\n🎯 Начать демонстрацию?")

    # Шаг 1: Описание задачи
    print_header("ШАГ 1: ОПИСАНИЕ ЗАДАЧИ", "=")

    print("📋 ПОСТАНОВКА ЗАДАЧИ:")
    print("   Необходимо предсказать, будет ли одобрена заявка на кредит")
    print("   на основе финансовых и персональных данных клиента\n")

    print("🎯 ТИП ЗАДАЧИ:")
    print("   Бинарная классификация (одобрено/отклонено)\n")

    print("📊 МЕТРИКА ОЦЕНКИ:")
    print("   ROC-AUC Score (площадь под ROC-кривой)\n")

    wait_for_enter()

    # Шаг 2: Загрузка данных
    print_header("ШАГ 2: ЗАГРУЗКА ДАННЫХ", "=")

    DATA_DIR = 'data/playground-series-s4e10'
    TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
    TEST_PATH = os.path.join(DATA_DIR, 'test.csv')

    if not os.path.exists(TRAIN_PATH):
        print("❌ ОШИБКА: Данные не найдены!")
        print(f"   Положите train.csv и test.csv в {DATA_DIR}/")
        return

    show_progress_bar("Загрузка данных с Kaggle...", 1.5)

    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

    print("✓ Данные успешно загружены!\n")
    print(f"   Train: {train_df.shape[0]:,} записей × {train_df.shape[1]} колонок")
    print(f"   Test:  {test_df.shape[0]:,} записей × {test_df.shape[1]} колонок\n")

    print("📊 ПЕРВЫЕ 5 ЗАПИСЕЙ TRAIN:")
    print(train_df.head().to_string())

    wait_for_enter()

    # Шаг 3: Анализ признаков
    print_header("ШАГ 3: АНАЛИЗ ПРИЗНАКОВ", "=")

    print("🔢 ЧИСЛОВЫЕ ПРИЗНАКИ (7):")
    numerical = ['person_age', 'person_income', 'person_emp_length',
                 'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                 'cb_person_cred_hist_length']
    for i, col in enumerate(numerical, 1):
        print(f"   {i}. {col}")

    print("\n📝 КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ (4):")
    categorical = ['person_home_ownership', 'loan_intent',
                   'loan_grade', 'cb_person_default_on_file']
    for i, col in enumerate(categorical, 1):
        print(f"   {i}. {col}")

    print("\n🎯 ЦЕЛЕВАЯ ПЕРЕМЕННАЯ:")
    print("   loan_status: 0 = без дефолта, 1 = дефолт")

    print("\n📈 РАСПРЕДЕЛЕНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ:")
    target_dist = train_df['loan_status'].value_counts()
    for label, count in target_dist.items():
        percent = (count / len(train_df)) * 100
        bar = "█" * int(percent / 2)
        print(f"   {label}: {count:,} ({percent:.1f}%) {bar}")

    wait_for_enter()

    # Шаг 4: Предобработка
    print_header("ШАГ 4: ПРЕДОБРАБОТКА ДАННЫХ", "=")

    show_progress_bar("Выполнение предобработки...", 1)

    X, y, X_test_raw = preprocess_data(train_df, test_df)

    print("✓ Предобработка завершена!\n")
    print("   Операции:")
    print("   • Удаление ID колонки")
    print("   • Отделение целевой переменной")
    print("   • Подготовка признаков для обучения\n")

    print(f"   X_train: {X.shape}")
    print(f"   y_train: {y.shape}")
    print(f"   X_test:  {X_test_raw.shape}")

    wait_for_enter()

    # Шаг 5: Разделение на train/validation
    print_header("ШАГ 5: РАЗДЕЛЕНИЕ НА TRAIN/VALIDATION", "=")

    show_progress_bar("Разделение данных (80/20)...", 1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("✓ Данные разделены!\n")
    print(f"   Train: {X_train.shape[0]:,} записей")
    print(f"   Valid: {X_val.shape[0]:,} записей")
    print(f"   Соотношение: 80% / 20%")
    print(f"   Stratified: да (сохранено распределение классов)")

    wait_for_enter()

    # Шаг 6: Трансформация признаков
    print_header("ШАГ 6: ТРАНСФОРМАЦИЯ ПРИЗНАКОВ", "=")

    show_progress_bar("Применение StandardScaler и OneHotEncoder...", 1.5)

    preprocessor = get_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test_raw)

    print("✓ Трансформация завершена!\n")
    print("   Числовые признаки:")
    print("   • SimpleImputer (median)")
    print("   • StandardScaler (нормализация)\n")
    print("   Категориальные признаки:")
    print("   • SimpleImputer (most_frequent)")
    print("   • OneHotEncoder (one-hot кодирование)\n")
    print(f"   Размерность после трансформации: {X_train_processed.shape[1]} признаков")

    wait_for_enter()

    # Шаг 7: Обучение модели
    print_header("ШАГ 7: ОБУЧЕНИЕ МОДЕЛИ", "=")

    print("🤖 МОДЕЛЬ: Gradient Boosting Classifier\n")
    print("⚙️ СЕТКА ПАРАМЕТРОВ GridSearchCV:")
    print("   • n_estimators: [50, 100, 200]")
    print("   • learning_rate: [0.01, 0.1, 0.2]")
    print("   • max_depth: [3, 5, 7]")
    print("   • Всего комбинаций: 27")
    print("   • Cross-validation: 3-fold")
    print("   • Метрика: roc_auc\n")

    wait_for_enter("⏳ ВНИМАНИЕ: Обучение займёт ~15-20 минут. Начать?")

    print("\n🚀 Запуск GridSearchCV...\n")
    print("Это может занять время. Идёт подбор лучших гиперпараметров...")
    print("(Выполняется 81 fit операция: 27 комбинаций × 3 фолда)\n")

    best_model, best_params = train_model(X_train_processed, y_train)

    print("\n✓ Обучение завершено!\n")
    print("🏆 ЛУЧШИЕ ПАРАМЕТРЫ:")
    for param, value in best_params.items():
        print(f"   • {param}: {value}")

    wait_for_enter()

    # Шаг 8: Оценка модели
    print_header("ШАГ 8: ОЦЕНКА МОДЕЛИ НА ВАЛИДАЦИИ", "=")

    show_progress_bar("Вычисление метрик...", 1)

    acc, auc = evaluate_model(best_model, X_val_processed, y_val)

    print("\n" + "═" * 80)
    print("🎯 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:".center(80))
    print("═" * 80)
    print(f"Validation Accuracy: {acc*100:.2f}%".center(80))
    print(f"Validation ROC-AUC:  {auc*100:.2f}%".center(80))
    print("═" * 80)

    wait_for_enter()

    # Шаг 9: Сохранение модели
    print_header("ШАГ 9: СОХРАНЕНИЕ МОДЕЛИ", "=")

    MODEL_PATH = 'best_model.pkl'

    show_progress_bar("Сохранение обученной модели...", 1)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)

    model_size = os.path.getsize(MODEL_PATH) / 1024  # KB

    print(f"✓ Модель сохранена: {MODEL_PATH}")
    print(f"   Размер: {model_size:.0f} KB")

    wait_for_enter()

    # Шаг 10: Генерация Submission
    print_header("ШАГ 10: ГЕНЕРАЦИЯ SUBMISSION ДЛЯ KAGGLE", "=")

    show_progress_bar("Генерация предсказаний для тестовой выборки...", 2)

    predictions = best_model.predict(X_test_processed)

    submission = pd.DataFrame({
        'id': test_df['id'],
        'loan_status': predictions
    })

    SUBMISSION_PATH = 'submission.csv'
    submission.to_csv(SUBMISSION_PATH, index=False)

    submission_size = os.path.getsize(SUBMISSION_PATH) / 1024  # KB

    print("✓ Submission создан!\n")
    print(f"   Файл: {SUBMISSION_PATH}")
    print(f"   Размер: {submission_size:.0f} KB")
    print(f"   Количество предсказаний: {len(submission):,}\n")

    print("📊 РАСПРЕДЕЛЕНИЕ ПРЕДСКАЗАНИЙ:")
    pred_dist = submission['loan_status'].value_counts()
    for label, count in pred_dist.items():
        percent = (count / len(submission)) * 100
        bar = "█" * int(percent / 2)
        status = "Одобрено" if label == 0 else "Отклонено"
        print(f"   {status} ({label}): {count:,} ({percent:.1f}%) {bar}")

    print("\n📋 ПЕРВЫЕ 10 ПРЕДСКАЗАНИЙ:")
    print(submission.head(10).to_string(index=False))

    wait_for_enter()

    # Финал
    print_header("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА", "=")

    print("✅ ВСЕ ЭТАПЫ УСПЕШНО ВЫПОЛНЕНЫ:\n")
    print("   1. ✓ Загрузка данных Kaggle")
    print("   2. ✓ Анализ признаков")
    print("   3. ✓ Предобработка данных")
    print("   4. ✓ Разделение на train/validation")
    print("   5. ✓ Трансформация признаков")
    print("   6. ✓ Обучение модели с GridSearchCV")
    print("   7. ✓ Оценка на валидации")
    print("   8. ✓ Сохранение модели")
    print("   9. ✓ Генерация submission для Kaggle\n")

    print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print(f"   • Accuracy:  {acc*100:.2f}%")
    print(f"   • ROC-AUC:   {auc*100:.2f}%")
    print(f"   • Модель:    {MODEL_PATH} ({model_size:.0f} KB)")
    print(f"   • Submission: {SUBMISSION_PATH} ({submission_size:.0f} KB)\n")

    print("🔗 ССЫЛКИ:")
    print("   • GitHub: https://github.com/Shchetnikovoff/loan_analitical_ML_model.git")
    print("   • Kaggle: https://www.kaggle.com/competitions/playground-series-s4e10\n")

    print("🎉 ПРОЕКТ ГОТОВ К ОТПРАВКЕ НА KAGGLE!")

    print("\n" + "=" * 80)
    print("Спасибо за внимание!".center(80))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Демонстрация прервана пользователем")
    except Exception as e:
        print(f"\n\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()

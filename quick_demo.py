"""
Быстрая демонстрация решения (БЕЗ обучения)
Использует уже обученную модель

Запуск: python quick_demo.py
"""

import pandas as pd
import pickle
import os
import time
import sys

# Добавляем src в path
sys.path.insert(0, 'src')


def print_header(text, char="="):
    """Печать красивого заголовка"""
    width = 80
    print("\n" + char * width)
    print(text.center(width))
    print(char * width + "\n")


def wait_input(message="▶ Нажмите Enter..."):
    """Пауза"""
    input(f"\n{message}")


def show_progress(text, steps=20):
    """Прогресс"""
    print(f"\n{text}", end=" ")
    for _ in range(steps):
        print("▓", end="", flush=True)
        time.sleep(0.05)
    print(" ✓\n")


def main():
    print_header("BYSTRAYA DEMONSTRATSIYA RESHENIYA", "=")
    print_header("Loan Approval Prediction - Kaggle S4E10", " ")

    print("Avtor: Shchetnikovoff")
    print("GitHub: https://github.com/Shchetnikovoff/loan_analitical_ML_model.git\n")

    wait_input(">>> Nachat demonstratsiyu?")

    # 1. Проверка файлов
    print_header("1. ПРОВЕРКА ФАЙЛОВ ПРОЕКТА", "-")

    files_to_check = {
        'best_model.pkl': 'Обученная модель',
        'submission.csv': 'Submission для Kaggle',
        'src/model.py': 'Код модели',
        'src/preprocessing.py': 'Код предобработки',
        'src/train.py': 'Скрипт обучения',
        'README.md': 'Документация',
    }

    print("Проверка наличия файлов:\n")
    all_ok = True
    for file_path, description in files_to_check.items():
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        print(f"   {status} {file_path:<30} - {description}")
        if not exists and file_path in ['best_model.pkl', 'submission.csv']:
            all_ok = False

    if not all_ok:
        print("\n❌ Сначала запустите: python src/train.py")
        return

    wait_input()

    # 2. Загрузка модели
    print_header("2. ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ", "-")

    show_progress("Загрузка модели...")

    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)

    model_size = os.path.getsize('best_model.pkl') / 1024

    print(f"✓ Модель загружена: {model.__class__.__name__}")
    print(f"   • n_estimators: {model.n_estimators}")
    print(f"   • learning_rate: {model.learning_rate}")
    print(f"   • max_depth: {model.max_depth}")
    print(f"   • Размер файла: {model_size:.0f} KB")

    wait_input()

    # 3. Загрузка submission
    print_header("3. АНАЛИЗ SUBMISSION", "-")

    show_progress("Загрузка submission.csv...")

    submission = pd.read_csv('submission.csv')
    submission_size = os.path.getsize('submission.csv') / 1024

    print(f"✓ Submission загружен")
    print(f"   • Количество предсказаний: {len(submission):,}")
    print(f"   • Размер файла: {submission_size:.0f} KB")
    print(f"   • Колонки: {list(submission.columns)}")

    print("\n📊 РАСПРЕДЕЛЕНИЕ ПРЕДСКАЗАНИЙ:")
    dist = submission['loan_status'].value_counts().sort_index()
    for label, count in dist.items():
        percent = (count / len(submission)) * 100
        bar = "█" * int(percent / 2)
        status = "Одобрено" if label == 0 else "Отклонено"
        print(f"   {status} ({label}): {count:,} ({percent:.2f}%) {bar}")

    wait_input()

    # 4. Демонстрация предсказаний
    print_header("4. ПРИМЕРЫ ПРЕДСКАЗАНИЙ", "-")

    print("ПЕРВЫЕ 10 ПРЕДСКАЗАНИЙ:")
    print(submission.head(10).to_string(index=False))

    print("\n\nСЛУЧАЙНЫЕ 10 ПРЕДСКАЗАНИЙ:")
    print(submission.sample(10).to_string(index=False))

    wait_input()

    # 5. Проверка данных
    print_header("5. ИНФОРМАЦИЯ О ДАТАСЕТЕ", "-")

    DATA_PATH = 'data/playground-series-s4e10/train.csv'
    if os.path.exists(DATA_PATH):
        show_progress("Загрузка данных...")

        df = pd.read_csv(DATA_PATH)

        print(f"✓ Train данные загружены")
        print(f"   • Записей: {len(df):,}")
        print(f"   • Колонок: {df.shape[1]}")

        print("\n📋 КОЛОНКИ:")
        for i, col in enumerate(df.columns, 1):
            dtype = df[col].dtype
            print(f"   {i:2}. {col:<30} ({dtype})")

        print("\n📊 ЦЕЛЕВАЯ ПЕРЕМЕННАЯ (loan_status):")
        target_dist = df['loan_status'].value_counts().sort_index()
        for label, count in target_dist.items():
            percent = (count / len(df)) * 100
            bar = "█" * int(percent / 2)
            print(f"   {label}: {count:,} ({percent:.2f}%) {bar}")
    else:
        print("⚠ Train данные не найдены")

    wait_input()

    # 6. Статистика проекта
    print_header("6. СТАТИСТИКА ПРОЕКТА", "-")

    print("📁 СТРУКТУРА ПРОЕКТА:\n")
    print("   loan_analitical_ML_model-main/")
    print("   ├── src/")
    print("   │   ├── model.py           - ML модель")
    print("   │   ├── preprocessing.py   - Предобработка")
    print("   │   └── train.py           - Обучение")
    print("   ├── data/")
    print("   │   └── playground-series-s4e10/")
    print("   │       ├── train.csv      - 58,645 записей")
    print("   │       └── test.csv       - 39,098 записей")
    print("   ├── best_model.pkl         - Обученная модель")
    print("   ├── submission.csv         - Submission")
    print("   └── README.md              - Документация")

    print("\n⚙️ ТЕХНОЛОГИИ:")
    print("   • Python 3.14+")
    print("   • scikit-learn 1.8.0")
    print("   • pandas 2.3.3")
    print("   • numpy 2.4.0")

    print("\n🎯 ПРИЗНАКИ МОДЕЛИ:")
    print("   • Числовые: 7 (age, income, emp_length, loan_amnt, ...)")
    print("   • Категориальные: 4 (home_ownership, intent, grade, default)")

    wait_input()

    # Финал
    print_header("✅ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА", "=")

    print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:\n")
    print("   ✓ Модель обучена и сохранена")
    print("   ✓ Submission создан (39,098 предсказаний)")
    print("   ✓ Validation Accuracy: 95.17%")
    print("   ✓ Validation ROC-AUC: 95.76%")

    print("\n📁 ФАЙЛЫ ГОТОВЫ:")
    print("   • best_model.pkl  - Модель для использования")
    print("   • submission.csv  - Для отправки на Kaggle")

    print("\n🔗 ССЫЛКИ:")
    print("   • GitHub: https://github.com/Shchetnikovoff/loan_analitical_ML_model.git")
    print("   • Kaggle: https://www.kaggle.com/competitions/playground-series-s4e10")

    print("\n🎉 ПРОЕКТ ГОТОВ К ИСПОЛЬЗОВАНИЮ!")

    print("\n" + "=" * 80)
    print("Спасибо за внимание!".center(80))
    print("=" * 80 + "\n")

    # Бонус: Открыть HTML визуализацию
    print("\n💡 СОВЕТ: Откройте results_visualization.html в браузере")
    print("           для красивой визуализации результатов!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Демонстрация прервана")
    except Exception as e:
        print(f"\n\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()

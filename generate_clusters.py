import pandas as pd
from sklearn.datasets import make_blobs
import os

def generate_dataset(n_samples, n_features, n_clusters, filename, cluster_std=1.0):
    """
    Генерирует датасет с кластерами и сохраняет его в CSV.
    """
    # Генерация данных
    # X - сами данные (признаки), y - метки кластеров
    X, y = make_blobs(
        n_samples=n_samples, 
        centers=n_clusters, 
        n_features=n_features, 
        cluster_std=cluster_std, # "Разброс" точек внутри кластера
        random_state=42          # Для воспроизводимости результата
    )
    
    # Формируем названия колонок: feature_1, feature_2, ...
    columns = [f"feature_{i+1}" for i in range(n_features)]
    
    # Создаем DataFrame
    df = pd.DataFrame(X, columns=columns)
    
    # Добавляем колонку с номером кластера
    df['cluster'] = y
    
    # Сохраняем в CSV
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Успешно! Датасет сохранен в файл: {os.path.abspath(filename)}")
    print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов.")
    print(f"Первые 5 строк:\n{df.head()}\n")

def main():
    print("="*40)
    print("  Генератор синтетического датасета")
    print("="*40)
    
    try:
        # Запрос параметров у пользователя
        n_clusters = int(input("Введите количество кластеров: "))
        n_features = int(input("Введите количество параметров (признаков) для одного объекта: "))
        
        samples_input = input("Введите общее количество объектов [по умолчанию 1000]: ")
        n_samples = int(samples_input) if samples_input.strip() else 1000
        
        filename_input = input("Введите имя выходного файла [по умолчанию dataset.csv]: ")
        filename = filename_input.strip() if filename_input.strip() else "dataset.csv"
        if not filename.endswith('.csv'):
            filename += '.csv'
            
        # Проверка на адекватность данных
        if n_clusters <= 0 or n_features <= 0 or n_samples <= 0:
            print("Ошибка: все значения должны быть положительными числами!")
            return
            
        if n_samples < n_clusters:
            print(f"Ошибка: количество объектов ({n_samples}) не может быть меньше количества кластеров ({n_clusters})!")
            return

        # Запуск генерации
        generate_dataset(n_samples, n_features, n_clusters, filename)

    except ValueError:
        print("Ошибка ввода: пожалуйста, вводите только целые числа!")

if __name__ == "__main__":
    main()
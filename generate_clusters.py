import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

def generate_dataset(n_samples, n_features, n_clusters, filename, cluster_std, save_labels):
    """
    Генерирует датасет с кластерами, сохраняет его в CSV и визуализирует.
    """
    print("Генерация данных...")
    X, y = make_blobs(
        n_samples=n_samples, 
        centers=n_clusters, 
        n_features=n_features, 
        cluster_std=cluster_std, 
        random_state=42
    )
    
    columns = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    
    # Добавляем метки только если пользователь этого хочет
    if save_labels:
        df['cluster'] = y
        print("Метки кластеров ДОБАВЛЕНЫ в файл (задача с учителем).")
    else:
        print("Метки кластеров СКРЫТЫ (задача для k-means без учителя).")
    
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Датасет сохранен в файл: {os.path.abspath(filename)}")
    print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов.")
    
    # Визуализируем всё равно (чтобы видеть "истинную" картину)
    visualize_clusters(X, y, n_features, n_clusters)
    
    return df

def visualize_clusters(X, y, n_features, n_clusters):
    """
    Визуализирует кластеры, адаптируясь к размерности данных.
    """
    print("Построение визуализации (истинное распределение)...")
    
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('default')
    
    if n_features == 2:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7, edgecolor='k')
        plt.xlabel('Параметр 1')
        plt.ylabel('Параметр 2')
        plt.title(f'Истинное распределение {n_clusters} кластеров (2D)')
        
    elif n_features == 3:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', alpha=0.7, edgecolor='k')
        ax.set_xlabel('Параметр 1')
        ax.set_ylabel('Параметр 2')
        ax.set_zlabel('Параметр 3')
        plt.title(f'Истинное распределение {n_clusters} кластеров (3D)')
        
    else:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7, edgecolor='k')
        
        var1 = pca.explained_variance_ratio_[0] * 100
        var2 = pca.explained_variance_ratio_[1] * 100
        
        plt.xlabel(f'Главная компонента 1 ({var1:.1f}% дисперсии)')
        plt.ylabel(f'Главная компонента 2 ({var2:.1f}% дисперсии)')
        plt.title(f'Истинное распределение {n_clusters} кластеров (PCA {n_features}D → 2D)')
    
    plt.colorbar(scatter, label='Истинный номер кластера')
    plt.tight_layout()
    print("Открывается окно с графиком. Закройте его, чтобы завершить работу скрипта.")
    plt.show()

def main():
    print("="*55)
    print("  Генератор датасетов для задач кластеризации")
    print("="*55)
    
    try:
        # 1. Базовые параметры
        n_clusters = int(input("Введите количество кластеров: "))
        n_features = int(input("Введите количество параметров (признаков): "))
        
        samples_input = input("Введите количество объектов [по умолчанию 1000]: ")
        n_samples = int(samples_input) if samples_input.strip() else 1000
        
        # 2. Разброс кластеров
        std_input = input("Введите разброс кластеров (стандартное отклонение) [по умолчанию 1.0]:\n"
                          "Подсказка: 0.5 - четкие кластеры, 2.0+ - сильно перемешаны: ")
        cluster_std = float(std_input) if std_input.strip() else 1.0
        
        if cluster_std <= 0:
            print("Ошибка: разброс должен быть положительным числом!")
            return

        # 3. Сохранять ли метки
        labels_input = input("Указывать номер кластера в CSV файле? (y/n) [по умолчанию y]: ").strip().lower()
        if labels_input == 'n' or labels_input == 'нет':
            save_labels = False
        else:
            save_labels = True
            
        # 4. Имя файла
        filename_input = input("Введите имя выходного файла [по умолчанию dataset.csv]: ")
        filename = filename_input.strip() if filename_input.strip() else "dataset.csv"
        if not filename.endswith('.csv'):
            filename += '.csv'
            
        # Валидация
        if n_clusters <= 0 or n_features <= 0 or n_samples <= 0:
            print("Ошибка: все базовые значения должны быть положительными числами!")
            return
            
        if n_samples < n_clusters:
            print(f"Ошибка: объектов ({n_samples}) не может быть меньше, чем кластеров ({n_clusters})!")
            return

        # Генерация
        generate_dataset(n_samples, n_features, n_clusters, filename, cluster_std, save_labels)

    except ValueError:
        print("Ошибка ввода: пожалуйста, вводите корректные числа!")

if __name__ == "__main__":
    main()
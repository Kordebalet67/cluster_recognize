import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

def generate_dataset(n_samples, n_features, n_clusters, filename, cluster_std=1.0):
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
    df['cluster'] = y
    
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Датасет сохранен в файл: {os.path.abspath(filename)}")
    print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов.")
    
    # Вызываем функцию визуализации
    visualize_clusters(X, y, n_features, n_clusters)
    
    return df

def visualize_clusters(X, y, n_features, n_clusters):
    """
    Визуализирует кластеры, адаптируясь к размерности данных.
    """
    print("Построение визуализации...")
    
    # Настраиваем стиль графиков
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    if n_features == 2:
        # 2D визуализация
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7, edgecolor='k')
        plt.xlabel('Параметр 1')
        plt.ylabel('Параметр 2')
        plt.title(f'Визуализация {n_clusters} кластеров (2D)')
        
    elif n_features == 3:
        # 3D визуализация
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', alpha=0.7, edgecolor='k')
        ax.set_xlabel('Параметр 1')
        ax.set_ylabel('Параметр 2')
        ax.set_zlabel('Параметр 3')
        plt.title(f'Визуализация {n_clusters} кластеров (3D)')
        
    else:
        # >3D: Используем PCA для снижения размерности до 2D
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7, edgecolor='k')
        
        # Добавляем информацию о том, сколько дисперсии сохранили главные компоненты
        var1 = pca.explained_variance_ratio_[0] * 100
        var2 = pca.explained_variance_ratio_[1] * 100
        
        plt.xlabel(f'Главная компонента 1 ({var1:.1f}% дисперсии)')
        plt.ylabel(f'Главная компонента 2 ({var2:.1f}% дисперсии)')
        plt.title(f'Визуализация {n_clusters} кластеров (PCA-проекция {n_features}D данных в 2D)')
    
    # Общие настройки для всех графиков
    plt.colorbar(scatter, label='Номер кластера')
    plt.tight_layout()
    plt.show()

def main():
    print("="*50)
    print("  Генератор и визуализатор синтетических кластеров")
    print("="*50)
    
    try:
        n_clusters = int(input("Введите количество кластеров: "))
        n_features = int(input("Введите количество параметров (признаков) для одного объекта: "))
        
        samples_input = input("Введите общее количество объектов [по умолчанию 1000]: ")
        n_samples = int(samples_input) if samples_input.strip() else 1000
        
        filename_input = input("Введите имя выходного файла [по умолчанию dataset.csv]: ")
        filename = filename_input.strip() if filename_input.strip() else "dataset.csv"
        if not filename.endswith('.csv'):
            filename += '.csv'
            
        if n_clusters <= 0 or n_features <= 0 or n_samples <= 0:
            print("Ошибка: все значения должны быть положительными числами!")
            return
            
        if n_samples < n_clusters:
            print(f"Ошибка: количество объектов ({n_samples}) не может быть меньше количества кластеров ({n_clusters})!")
            return

        generate_dataset(n_samples, n_features, n_clusters, filename)
        print("Готово! Работа скрипта завершена.")

    except ValueError:
        print("Ошибка ввода: пожалуйста, вводите только целые числа!")

if __name__ == "__main__":
    main()
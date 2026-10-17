import os
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42  # Для воспроизводимости разбиения

# Типичные названия колонок с метками.
LABEL_ALIASES = {
    "cluster", "clusters", "cluster_id", "cluster_label",
    "label", "labels", "target", "y", "class",
    "кластер", "метка", "класс", "цель",
}


def load_dataset(path):
    """Загружает CSV и проверяет, что в нём есть данные."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Файл не найден: {os.path.abspath(path)}")

    df = pd.read_csv(path)

    if df.empty or df.shape[1] == 0:
        raise ValueError("Файл пуст или не содержит колонок с данными.")

    return df


def find_label_column(df):
    """Ищет колонку с метками по имени. Возвращает имя или None."""
    matches = [
        c for c in df.columns
        if str(c).strip().lower() in LABEL_ALIASES
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        print(f"Найдено несколько подходящих колонок: {matches}")
        while True:
            answer = input("Какую считать меткой кластера? "
                           "(Enter — не удалять ничего): ").strip()
            if not answer:
                return None
            if answer in df.columns:
                return answer
            print("Колонки с таким именем нет.")

    # Совпадений нет: либо колонка называется иначе,
    # либо датасет уже был сгенерирован без меток.
    print("Стандартная колонка с метками не найдена.")
    print(f"Доступные колонки: {list(df.columns)}")

    while True:
        answer = input("Укажите имя колонки с метками "
                       "(Enter — меток в файле нет): ").strip()
        if not answer:
            return None
        if answer in df.columns:
            return answer
        print("Колонки с таким именем нет.")


def split_dataset(df, label_col=None, method="shuffle", random_state=RANDOM_STATE):
    """
    Делит DataFrame на две СТРОГО равные части и удаляет колонку меток.

    method:
        "sequential" — по порядку строк (без перемешивания);
        "shuffle"    — случайное перемешивание (рекомендуется);
        "stratified" — перемешивание с сохранением пропорций кластеров
                       (использует label_col ТОЛЬКО на момент разбиения).

    Возвращает: (first, second, dropped_count)
    """
    if len(df) < 2:
        raise ValueError("Нужно минимум 2 объекта для деления на две части.")

    if label_col is not None and len(df.columns) == 1:
        raise ValueError("В файле только колонка меток — после её удаления "
                         "не останется признаков.")

    # 1. Формируем порядок строк.
    if method == "stratified" and label_col is not None:
        try:
            part_a, part_b = train_test_split(
                df,
                test_size=0.5,
                random_state=random_state,
                stratify=df[label_col],
            )
            # concat сохраняет разделение: первая половина = part_a,
            # вторая половина = part_b.
            ordered = pd.concat([part_a, part_b])
        except ValueError:
            print("Стратификация невозможна (есть кластер с <2 объектами "
                  "или пропуски в метках). Перехожу к обычному перемешиванию.")
            ordered = df.sample(frac=1, random_state=random_state)

    elif method == "shuffle":
        ordered = df.sample(frac=1, random_state=random_state)

    else:  # sequential
        ordered = df.copy()

    ordered = ordered.reset_index(drop=True)

    # 2. Делаем число строк чётным, чтобы части были строго равны.
    dropped = 0
    if len(ordered) % 2 != 0:
        # Объект удаляется уже ПОСЛЕ перемешивания, т.е. случайно.
        ordered = ordered.iloc[:-1].copy()
        dropped = 1

    # 3. Ровно пополам.
    half = len(ordered) // 2
    first = ordered.iloc[:half].copy()
    second = ordered.iloc[half:].copy()

    # 4. Удаляем истинные метки из обеих частей.
    if label_col is not None and label_col in first.columns:
        first = first.drop(columns=[label_col])
        second = second.drop(columns=[label_col])

    return (
        first.reset_index(drop=True),
        second.reset_index(drop=True),
        dropped,
    )


def ask_output_name(prompt, default, input_path, other_name=None):
    """Запрашивает имя выходного файла и проверяет его на конфликты."""
    while True:
        raw = input(prompt).strip() or default
        name = raw if raw.lower().endswith(".csv") else raw + ".csv"
        abs_name = os.path.abspath(name)

        if abs_name == os.path.abspath(input_path):
            print("Нельзя перезаписывать исходный датасет.")
            continue

        if other_name and abs_name == os.path.abspath(other_name):
            print("Это имя уже занято другой частью.")
            continue

        if os.path.exists(abs_name):
            answer = input(
                f"Файл '{name}' уже существует. Перезаписать? (y/n) [n]: "
            ).strip().lower()
            if answer not in ("y", "yes", "д", "да"):
                continue

        return name


def main():
    print("=" * 60)
    print("  Разделение датасета на две равные части")
    print("=" * 60)

    try:
        # --- 1. Загрузка ---
        raw_path = input("Путь к исходному CSV [dataset.csv]: ").strip()
        input_path = raw_path or "dataset.csv"

        df = load_dataset(input_path)
        print(f"Загружено: {df.shape[0]} объектов × {df.shape[1]} колонок")
        print(f"   Колонки: {list(df.columns)}")

        # --- 2. Поиск меток ---
        label_col = find_label_column(df)
        if label_col:
            print(f"Метки найдены в колонке '{label_col}' — она будет удалена.")
        else:
            print("Меток в файле нет — датасет уже 'чистый'.")

        # --- 3. Способ деления ---
        print("Как делить данные?")
        print("  1. Sequential — по порядку строк (НЕ рекомендуется)")
        print("  2. Shuffle    — случайное перемешивание [рекомендуется]")
        print("  3. Stratified — случайно + одинаковые пропорции кластеров")
        choice = input("Ваш выбор (1/2/3) [2]: ").strip() or "2"

        if choice == "1":
            method = "sequential"
        elif choice == "3":
            if label_col is None:
                print("Нет колонки с метками — stratified недоступен. "
                      "Использую shuffle.")
                method = "shuffle"
            else:
                method = "stratified"
        else:
            method = "shuffle"

        # --- 4. Проверка чётности ---
        if len(df) < 2:
            print("В датасете меньше двух объектов — делить нечего.")
            return

        if len(df) % 2 != 0:
            print(f"Нечётное число объектов: {len(df)}.")
            answer = input(
                "Отбросить один объект, чтобы части были строго равны? "
                "(y/n) [y]: "
            ).strip().lower()
            if answer in ("n", "нет", "no"):
                print("Отменено. Сгенерируйте датасет с чётным числом объектов.")
                return

        # --- 5. Имена частей ---
        print("Задайте названия двух частей (это будут имена CSV-файлов).")
        name_1 = ask_output_name("Название первой части [part_1]: ",
                                 "part_1", input_path)
        name_2 = ask_output_name("Название второй части [part_2]: ",
                                 "part_2", input_path, name_1)

        # --- 6. Разделение и сохранение ---
        first, second, dropped = split_dataset(df, label_col, method)

        first.to_csv(name_1, index=False, encoding="utf-8")
        second.to_csv(name_2, index=False, encoding="utf-8")

        # --- 7. Отчёт ---
        print("\n" + "=" * 60)
        print("Готово!")
        print(f"   Способ деления : {method}")
        print(f"   Random state   : {RANDOM_STATE}")
        if dropped:
            print(f"   Отброшено      : {dropped} объект (нечётное исходное число)")
        if label_col:
            print(f"   Удалена колонка: '{label_col}'")

        print(f"\n   Часть 1 → {os.path.abspath(name_1)}")
        print(f"             {len(first)} строк × {len(first.columns)} колонок")
        print(f"   Часть 2 → {os.path.abspath(name_2)}")
        print(f"             {len(second)} строк × {len(second.columns)} колонок")

        assert len(first) == len(second), "Части получились неравными!"

        print(f"Предпросмотр первой части:\n"
              f"{first.head(3).to_string(index=False)}")
        print("Исходный файл не изменён: в нём остались истинные метки. "
              "Они пригодятся для оценки качества кластеризации (ARI, silhouette).")

    except FileNotFoundError as e:
        print(f"{e}")
    except pd.errors.EmptyDataError:
        print("Файл пуст или не является корректным CSV.")
    except ValueError as e:
        print(f"Ошибка данных: {e}")
    except KeyboardInterrupt:
        print("Отменено пользователем.")


if __name__ == "__main__":
    main()
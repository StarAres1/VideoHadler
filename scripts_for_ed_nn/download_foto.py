import requests
import os
from tqdm import tqdm

dir_name = "../dataset_RAISE"
os.makedirs(dir_name, exist_ok=True)
counter = 0
skipped = 0

with open("../RAISE_2k.csv", "r") as file:
    # Пропускаем заголовок, если он есть
    first_line = file.readline().strip()

    # Сначала подсчитаем общее количество строк для прогресс-бара
    file.seek(0)
    file.readline()  # пропускаем заголовок
    total_lines = sum(1 for _ in file)

    # Возвращаемся в начало файла
    file.seek(0)
    file.readline()  # пропускаем заголовок

    with tqdm(total=total_lines, desc="Скачивание", unit="файл") as pbar:
        for line in file:
            try:
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue

                file_name, _, file_url = parts[:3]

                # Формируем полный путь к файлу
                full_file_path = os.path.join(dir_name, file_name + ".tiff")

                # Проверяем, существует ли уже файл
                if os.path.exists(full_file_path):
                    # print(f"Файл {file_name}.tiff уже существует, пропускаем")
                    skipped += 1
                    pbar.update(1)
                    continue

                # Скачиваем файл, если он не существует
                response = requests.get(file_url, stream=True, timeout=30)
                response.raise_for_status()

                with open(full_file_path, "wb") as image:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            image.write(chunk)

                counter += 1
                pbar.update(1)
                pbar.set_postfix({"Скачано": counter, "Пропущено": skipped})

            except requests.exceptions.RequestException as e:
                print(f"\nОшибка при скачивании {file_name}: {e}")
            except Exception as ee:
                print(f"\nОшибка: {ee}")

print(f"\nЗавершено. Скачано: {counter}, Пропущено (уже есть): {skipped}")
import os
import cv2
import numpy as np
from tqdm import tqdm

def process_tiff_images(input_folder, output_folder, scale_factor):
    """
    Обрабатывает все TIFF-изображения в папке:
    - переименовывает в порядковый номер (1,2,3...)
    - уменьшает разрешение в scale_factor раз по горизонтали и вертикали
    - сохраняет в формате PNG в выходную папку

    :param input_folder: путь к папке с исходными TIFF-файлами
    :param output_folder: путь для сохранения обработанных PNG
    :param scale_factor: во сколько раз уменьшить изображение (целое число > 1)
    """
    # Создаём выходную папку, если её нет
    os.makedirs(output_folder, exist_ok=True)

    # Получаем список всех файлов в исходной папке с расширением .tiff или .tif
    all_files = os.listdir(input_folder)
    tiff_files = [f for f in all_files if f.lower().endswith(('.tiff', '.tif'))]

    # Подсчёт количества файлов (можно просто len(tiff_files))
    total_files = len(tiff_files)
    print(f"Найдено TIFF-файлов: {total_files}")

    if total_files == 0:
        print("Нет файлов для обработки.")
        return

    # Перебираем файлы с индексацией от 1 и прогресс-баром
    for idx, filename in enumerate(tqdm(tiff_files, desc="Обработка изображений"), start=1):
        input_path = os.path.join(input_folder, filename)

        # Чтение изображения с помощью OpenCV
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Предупреждение: не удалось прочитать {filename}. Пропускаем.")
            continue

        # Вычисление нового размера
        height, width = img.shape[:2]
        new_width = int(width // scale_factor)
        new_height = int(height // scale_factor)

        # Уменьшение размера (интерполяция по умолчанию INTER_LINEAR подходит)
        resized_img = cv2.resize(img, (new_width, new_height))

        # Формирование имени выходного файла (порядковый номер)
        output_filename = f"{idx}.png"
        output_path = os.path.join(output_folder, output_filename)

        # Сохранение в PNG
        cv2.imwrite(output_path, resized_img)

    print(f"Готово! Обработанные изображения сохранены в: {output_folder}")

# Пример использования
if __name__ == "__main__":
    # Задайте свои параметры
    input_directory = "../dataset_RAISE"   # папка с исходными TIFF
    output_directory = "../dataset"   # папка для результатов
    scale = 2                            # уменьшение в 2 раза

    process_tiff_images(input_directory, output_directory, scale)
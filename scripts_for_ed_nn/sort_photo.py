import json
import os
import shutil

# ================= НАСТРОЙКИ =================
# Путь к файлу с метаданными
JSONL_PATH = '/scripts_for_ed_nn/results_markup/best_methods_baseline_filter(huge_ssim_6)1.jsonl'

# Папка, где сейчас лежат исходные фотографии
SOURCE_IMAGES_DIR = 'C:/Projects/QtTrial/dataset/half'

# Папка, куда будут перемещены фотографии (создастся автоматически)
DEST_ROOT_DIR = 'C:/Projects/QtTrial/sorted'

# Имя выходного файла с картой папок
OUTPUT_JSON_PATH = 'results_markup/final.json'


# =============================================

def get_folder_name(method, params):
    """
    Формирует имя папки на основе метода и отсортированных параметров.
    Пример: adjust_contrast_2.0_6
    """
    # Сортируем параметры по ключам, чтобы порядок был всегда одинаковым
    # (например, alpha всегда перед beta)
    sorted_params = sorted(params.items())

    # Преобразуем значения в строку.
    # float(2.0) станет "2.0", int(6) станет "6"
    params_str = "_".join([str(v) for k, v in sorted_params])

    return f"{method}_{params_str}"


def find_image_file(source_dir, image_name):
    """
    Ищет файл изображения.
    Сначала пробует точное совпадение, затем добавляет популярные расширения.
    """
    # 1. Пробуем найти файл точно так, как написано в JSON
    full_path = os.path.join(source_dir, image_name)
    if os.path.isfile(full_path):
        return full_path

    # 2. Если не нашли, пробуем добавить расширения (png, jpg, jpeg)
    # Часто в JSON имя без расширения, а на диске оно есть
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    for ext in extensions:
        path_with_ext = os.path.join(source_dir, image_name + ext)
        if os.path.isfile(path_with_ext):
            return path_with_ext

        # Также проверяем, вдруг в JSON уже есть расширение, но мы ищем без него
        # (на случай если image_name уже содержит .png, но мы добавили еще одно)
        # Этот блок нужен редко, но для надежности можно проверить совпадение по началу имени
        # Но в рамках задачи логичнее предположить, что в JSON нет расширения.

    # 3. Если в JSON имя уже с расширением, но файл не найден выше
    # (на случай если в JSON "image.png", а мы искали "image.png" в шаге 1 - это сработает)
    # Дополнительная проверка: если в image_name нет точки, мы уже добавили расширения.
    # Если точка есть, возможно файл просто отсутствует.

    return None


def main():
    # Создаем корневую папку для результатов, если нет
    if not os.path.exists(DEST_ROOT_DIR):
        os.makedirs(DEST_ROOT_DIR)

    # Список для хранения уникальных имен созданных папок
    # Используем список, чтобы сохранить порядок первого создания
    unique_folders = []

    # Множество для быстрой проверки существования папки в списке
    unique_folders_set = set()

    print(f"Читаю файл {JSONL_PATH}...")

    try:
        with open(JSONL_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Ошибка: Файл {JSONL_PATH} не найден.")
        return

    processed_count = 0
    error_count = 0

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            print(f"Ошибка JSON в строке {line_num}: {line}")
            error_count += 1
            continue

        image_name = data.get('image')
        method = data.get('method')
        params = data.get('params', {})

        if not image_name or not method:
            print(f"Пропущена строка {line_num}: нет имени изображения или метода.")
            continue

        # 1. Формируем имя целевой папки
        folder_name = get_folder_name(method, params)
        dest_folder_path = os.path.join(DEST_ROOT_DIR, folder_name)

        # 2. Добавляем папку в список уникальных, если она еще не встречалась
        if folder_name not in unique_folders_set:
            unique_folders.append(folder_name)
            unique_folders_set.add(folder_name)
            # Создаем папку физически
            os.makedirs(dest_folder_path, exist_ok=True)

        # 3. Ищем исходный файл
        source_file_path = find_image_file(SOURCE_IMAGES_DIR, image_name)

        if source_file_path:
            # 4. Перемещаем файл
            dest_file_path = os.path.join(dest_folder_path, os.path.basename(source_file_path))
            try:
                shutil.move(source_file_path, dest_file_path)
                processed_count += 1
                # print(f"Перемещено: {image_name} -> {folder_name}") # Можно раскомментировать для лога
            except Exception as e:
                print(f"Ошибка при перемещении {image_name}: {e}")
                error_count += 1
        else:
            print(f"Внимание: Файл для '{image_name}' не найден в папке {SOURCE_IMAGES_DIR}")
            error_count += 1

    # 5. Формируем итоговый JSON
    # Структура: список объектов с индексом и именем папки
    folder_map_data = []
    for index, folder_name in enumerate(unique_folders, start=1):
        folder_map_data.append({
            "id": index,
            "folder_name": folder_name
        })

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as out_f:
        json.dump(folder_map_data, out_f, indent=2, ensure_ascii=False)

    print("-" * 30)
    print("Работа завершена.")
    print(f"Обработано файлов: {processed_count}")
    print(f"Ошибок/Пропусков: {error_count}")
    print(f"Создано уникальных папок: {len(unique_folders)}")
    print(f"Отчет сохранен в: {OUTPUT_JSON_PATH}")


if __name__ == '__main__':
    main()
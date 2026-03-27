import os
import cv2

def process_single_image_with_all_interpolations(input_path, output_folder, scale_factor):
    """
    Принимает одно изображение, уменьшает его в scale_factor раз,
    применяя все возможные методы интерполяции OpenCV,
    и сохраняет результаты в выходную папку.

    :param input_path: путь к исходному изображению (поддерживаются форматы, читаемые OpenCV)
    :param output_folder: папка для сохранения результатов
    :param scale_factor: во сколько раз уменьшить изображение (целое число > 1)
    """
    # Создаём выходную папку, если её нет
    os.makedirs(output_folder, exist_ok=True)

    # Извлекаем имя файла без расширения для формирования имён результатов
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # Чтение изображения (сохраняем все каналы, включая альфа-канал, если есть)
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Ошибка: не удалось прочитать изображение {input_path}")
        return

    # Вычисляем новый размер после уменьшения
    h, w = img.shape[:2]
    new_w = int(w // scale_factor)
    new_h = int(h // scale_factor)

    # Словарь доступных методов интерполяции в OpenCV
    interpolation_methods = {
        'INTER_NEAREST': cv2.INTER_NEAREST,
        'INTER_LINEAR': cv2.INTER_LINEAR,
        'INTER_CUBIC': cv2.INTER_CUBIC,
        'INTER_AREA': cv2.INTER_AREA,
        'INTER_LANCZOS4': cv2.INTER_LANCZOS4,
        'INTER_LINEAR_EXACT': cv2.INTER_LINEAR_EXACT,
        'INTER_NEAREST_EXACT': cv2.INTER_NEAREST_EXACT,
        # Добавьте другие методы, если они есть в вашей версии OpenCV
    }

    print(f"Обработка изображения '{base_name}' с уменьшением в {scale_factor} раза...")
    for method_name, method_flag in interpolation_methods.items():
        try:
            # Изменение размера с указанным методом интерполяции
            resized = cv2.resize(img, (new_w, new_h), interpolation=method_flag)

            # Формируем имя выходного файла: исходное_имя_метод_scale.png
            output_filename = f"{base_name}_{method_name}_scale{scale_factor}.png"
            output_path = os.path.join(output_folder, output_filename)

            # Сохраняем результат в формате PNG
            cv2.imwrite(output_path, resized)
            print(f"  ✓ Сохранено: {output_filename}")
        except Exception as e:
            print(f"  ✗ Ошибка при использовании {method_name}: {e}")

    print(f"\nГотово! Все результаты сохранены в папке: {output_folder}")

# Пример использования
if __name__ == "__main__":
    # Укажите путь к одному конкретному файлу
    input_image = "../dataset_RAISE/r000da54ft.tiff"   # замените на ваш файл
    output_directory = "../output_interpolations"    # папка для результатов
    scale = 2                                         # коэффициент уменьшения

    process_single_image_with_all_interpolations(input_image, output_directory, scale)
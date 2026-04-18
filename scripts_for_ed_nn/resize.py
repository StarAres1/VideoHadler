import os
from PIL import Image
from tqdm import tqdm

def resize_images_to_half(input_dir='.', output_dir='C:\Projects\QtTrial\dataset\datase\half_ref'):
    """
    Уменьшает все PNG изображения в указанной папке в 2 раза по ширине и высоте.
    Результаты сохраняются в отдельную папку (по умолчанию 'half').
    Отображает прогресс с помощью tqdm.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        tqdm.write(f"Создана папка для результатов: {output_dir}")

    # Собираем список PNG-файлов
    png_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]

    if not png_files:
        tqdm.write("PNG файлы не найдены.")
        return

    # Обрабатываем с индикатором прогресса
    for filename in tqdm(png_files, desc="Обработка изображений", unit="file"):
        input_path = os.path.join(input_dir, filename)
        try:
            with Image.open(input_path) as img:
                original_width, original_height = img.size
                new_width = original_width // 2
                new_height = original_height // 2

                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                output_path = os.path.join(output_dir, filename)
                resized_img.save(output_path)
        except Exception as e:
            tqdm.write(f"Ошибка при обработке {filename}: {e}")

if __name__ == "__main__":
    resize_images_to_half("C:\Projects\QtTrial\dataset\dataset")
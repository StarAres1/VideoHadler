import os
import shutil
from pathlib import Path

# ===== Настройки =====
SOURCE_DIR = r"organized_images"   # укажите ваш путь
TARGET_DIR = r"../../dataset/half"   # укажите ваш путь
# ====================

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg'}

def collect_images(source_dir: Path):
    image_paths = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                image_paths.append(Path(root) / file)
    return image_paths

def move_image(src: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / src.name
    if dest_path.exists():
        base = src.stem
        ext = src.suffix
        counter = 1
        while True:
            new_name = f"{base}_{counter}{ext}"
            dest_path = dest_dir / new_name
            if not dest_path.exists():
                break
            counter += 1
        print(f"Конфликт: {src.name} -> {new_name}")
    shutil.move(str(src), str(dest_path))
    print(f"Перемещён: {src} -> {dest_path}")

def main():
    source = Path(SOURCE_DIR).resolve()
    target = Path(TARGET_DIR).resolve()
    if not source.is_dir():
        print(f"Ошибка: исходная директория '{source}' не существует.")
        return
    print(f"Поиск изображений в: {source}")
    images = collect_images(source)
    if not images:
        print("Изображений не найдено.")
        return
    print(f"Найдено изображений: {len(images)}")
    for img in images:
        move_image(img, target)
    print("Готово!")

if __name__ == "__main__":
    main()
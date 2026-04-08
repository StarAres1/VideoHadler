import cv2
import os
from app.core.QualityImprover import QualityImprover
from app.core.ContrastImprover import ContrastImprover

def apply_methods_to_image(image_path, output_dir="results"):
    # Создаем директорию для результатов, если её нет
    os.makedirs(output_dir, exist_ok=True)

    # Получаем имя файла без расширения
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Загружаем изображение
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return

    print("Original shape:", img.shape)
    print(f"\nОбработка изображения: {image_path}")

    # Словарь методов для применения
    methods = [
        # Методы улучшения контраста
        ("CLAHE", ContrastImprover.CLAHE, {"clipLimit": 4.0, "titleGridSizeX": 8, "titleGridSizeY": 8}),
        ("HE", ContrastImprover.HE, {}),
        ("Retinex", ContrastImprover.adjust_contrast, {"alpha": 2, "beta": 10}),
        ("gamma_correction", ContrastImprover.gamma_correction, {"gamma": 1.5}),
        ("sigmoid_correction", ContrastImprover.sigmoid_correction, {"cutoff": 0.5, "gain": 10}),
        ("auto_gamma", ContrastImprover.auto_gamma, {}),
        ("combined_enhancement", ContrastImprover.combined_enhancement, {"clip_limit": 4.0, "sigmoid_gain": 8}),

        # Методы улучшения качества
        ("blur", QualityImprover.blur, {}),
        ("gaussianBlur", QualityImprover.gaussianBlur, {}),
        ("medianBlur", QualityImprover.medianBlur, {}),
        ("bilateralFilter", QualityImprover.bilateralFilter, {}),
        ("fastNl", QualityImprover.fastNl, {}),
    ]

    # Применяем каждый метод
    for method_name, method_func, params in methods:
        try:
            # Формируем строку с параметрами для имени файла
            param_str = ""
            if params:
                param_items = [f"{k}={v}" for k, v in params.items()]
                param_str = "_" + ",".join(param_items)

            # Применяем метод
            if params:
                result = method_func(img, **params)
            else:
                result = method_func(img)

            if result is not None:
                # Сохраняем результат
                output_filename = f"{base_name}_{method_name}{param_str}.tiff"
                output_path = os.path.join(output_dir, output_filename)

                # Конвертируем RGB обратно в BGR для сохранения
                if len(result.shape) == 3 and result.shape[2] == 3:
                    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(output_path, result_bgr)
                else:
                    cv2.imwrite(output_path, result)

                print(f"  ✓ {method_name} -> {output_filename}")
            else:
                print(f"  ✗ {method_name} -> вернул None")

        except Exception as e:
            print(f"  ✗ Ошибка в {method_name}: {str(e)}")

def main():
    # Список изображений для обработки
    images = ["1.tiff"]

    # Проверяем существование файлов
    for img_path in images:
        if not os.path.exists(img_path):
            print(f"Предупреждение: файл {img_path} не найден!")

    # Применяем методы к каждому изображению
    for img_path in images:
        if os.path.exists(img_path):
            apply_methods_to_image(img_path)
        else:
            print(f"Пропускаем {img_path} (файл не найден)")


if __name__ == "__main__":
    main()
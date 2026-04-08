import cv2
import numpy as np
import os
from tqdm import tqdm

def contrast_change_gray(image, alpha):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    distorted = (1 - alpha) * image.astype(np.float32) + alpha * gray_3ch.astype(np.float32)
    distorted = np.clip(distorted, 0, 255).astype(np.uint8)

    return distorted


def contrast_change(image, alpha):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Применяем контраст только к каналу яркости (L)
    l = cv2.convertScaleAbs(l, alpha=alpha, beta=0)

    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return result

def gamma_transfer(image, gamma):
    normalized = image.astype(np.float32) / 255.0
    corrected = np.power(normalized, gamma)
    result = (corrected * 255).astype(np.uint8)

    return result


def cubic_distortion(image, a1, a2, a3, a4):
    x = image.astype(np.float32) / 255.0
    y = a1 * (x ** 3) + a2 * (x ** 2) + a3 * x + a4
    y = np.clip(y, 0, 1)
    return (y * 255).astype(np.uint8)


def logistic_distortion(image, p1, p2, p3, p4):
    """
    Применяет логистическое искажение к каналу L в LAB.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)

    # Нормализуем L в [0, 1]
    l_norm = l / 255.0
    l_new = (p1 - p2) / (1 + np.exp(-(l_norm - p3) / p4)) + p2
    l_new = np.clip(l_new, 0, 1)
    l_new = (l_new * 255).astype(np.float32)

    lab_dist = cv2.merge([l_new, a, b])
    bgr_dist = cv2.cvtColor(lab_dist.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return bgr_dist


def mean_shift(image, delta):

    shifted = image.astype(np.int16) + delta
    shifted = np.clip(shifted, 0, 255).astype(np.uint8)
    return shifted


img = cv2.imread('../2.tiff')

logistic_params = [
    (1.0, 0.0, 0.6, 0.15),
    (0.9, 0.1, 0.5, 0.05)
]

cubic_params = [
    (2.0, -3.0, 2.0, 0.0),
    (-2.0, 3.0, 0.0, 0.0),
    (0.0, 0.0, 3.0, -0.5),
    (-3.0, 4.0, -1.0, 0.8)
]

deltas = [-120, -100, -80, -60, -40, -20, 20, 40, 60, 80, 100, 120]
contrast_alphas = [0.5, 0.75, 1.0, 1.25, 1.5]
contrast_gray_alphas = [0.5, 0.75]
gammas = [1/5, 1/3, 1/2, 1/1.5, 1.5, 2, 3, 5]

# Тесты на одной картинке
"""
# Использование всех 12 уровней
deltas = [-120, -100, -80, -60, -40, -20, 20, 40, 60, 80, 100, 120]
for d in deltas:
    degraded = mean_shift(img, d)
    cv2.imwrite(f'../result_contrast_change/meanshift_{d}.jpg', degraded)


for i in [0.5, 0.75, 1.0, 1.25, 1.5]:
    distorted = contrast_change(img, i)
    cv2.imwrite(f'../result_contrast_change/contrast_change_alpha_{i}.tiff', distorted)

for i in [0.5, 0.75]:
    distorted = contrast_change_gray(img, i)
    cv2.imwrite(f'../result_contrast_change/contrast_change_gray_{i}.tiff', distorted)

gammas = [1/5, 1/3, 1/2, 1/1.5, 1.5, 2, 3, 5]
for g in gammas:
    degraded = gamma_transfer(img, g)
    cv2.imwrite(f'../result_contrast_change/gamma_{g}.tiff', degraded)
    
for i, (a1, a2, a3, a4) in enumerate(cubic_params):
    degraded = cubic_distortion(img, a1, a2, a3, a4)
    cv2.imwrite(f'../result_contrast_change/cubic_level_{i+1}.jpg', degraded)
    
for i, (p1, p2, p3, p4) in enumerate(logistic_params):
    degraded = logistic_distortion(img, p1, p2, p3, p4)
    cv2.imwrite(f'../result_contrast_change/logistic_level_{i+1}.jpg', degraded)
"""

# Список заданий: (функция, список параметров, формат имени)
# Для cubic и logistic параметры — кортежи, поэтому будем использовать индекс + 1 как уровень
distortion_tasks = [
    (mean_shift, deltas, "meanshift_{}"),
    (contrast_change, contrast_alphas, "contrast_change_alpha_{}"),
    (contrast_change_gray, contrast_gray_alphas, "contrast_change_gray_{}"),
    (gamma_transfer, gammas, "gamma_{}"),
    (cubic_distortion, cubic_params, "cubic_level_{}"),
    (logistic_distortion, logistic_params, "logistic_level_{}")
]

# ==================== Основная функция ====================
def process_all(input_folder, distorted_folder, start_from=1):
    """
    Применяет искажения ко всем PNG-файлам из input_folder,
    начиная с файла с номером start_from (включительно).

    :param input_folder:   папка с исходными PNG (имена: число.png)
    :param distorted_folder: папка для сохранения искажённых PNG
    :param start_from:     номер первого обрабатываемого файла (по умолчанию 1)
    """
    # Получаем список всех PNG-файлов
    png_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]
    # Сортируем по числовому значению имени (без расширения)
    png_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    if not png_files:
        print("Нет PNG-файлов для искажения.")
        return

    # Определяем минимальный и максимальный номера в папке
    indices = [int(os.path.splitext(f)[0]) for f in png_files]
    min_idx, max_idx = min(indices), max(indices)
    print(f"Доступны изображения с номерами от {min_idx} до {max_idx}")

    # Фильтруем файлы, начиная с указанного номера
    files_to_process = [f for f in png_files if int(os.path.splitext(f)[0]) >= start_from]

    if not files_to_process:
        print(f"Нет изображений с номером >= {start_from}. Работа завершена.")
        return

    first_processed = int(os.path.splitext(files_to_process[0])[0])
    last_processed = int(os.path.splitext(files_to_process[-1])[0])
    print(f"Будут обработаны номера с {first_processed} по {last_processed} (всего {len(files_to_process)} файлов)")

    # Подсчёт операций
    ops_per_image = sum(len(params) for _, params, _ in distortion_tasks)
    total_ops = len(files_to_process) * ops_per_image

    os.makedirs(distorted_folder, exist_ok=True)

    print(f"Начинаем искажения: {len(files_to_process)} изображений × {ops_per_image} операций = {total_ops} файлов.")
    with tqdm(total=total_ops, desc="Искажения", unit="img") as pbar:
        for png_file in files_to_process:
            idx = os.path.splitext(png_file)[0]  # номер как строка
            img_path = os.path.join(input_folder, png_file)
            img = cv2.imread(img_path)
            if img is None:
                tqdm.write(f"Ошибка чтения {png_file}, пропускаем.")
                pbar.update(ops_per_image)  # пропускаем весь файл целиком
                continue

            # Перебираем все виды искажений
            for func, params, name_template in distortion_tasks:
                if func in (cubic_distortion, logistic_distortion):
                    # Для функций с несколькими параметрами (кортежами)
                    for i, param_set in enumerate(params, start=1):
                        if func is cubic_distortion:
                            a1, a2, a3, a4 = param_set
                            distorted = func(img, a1, a2, a3, a4)
                        else:  # logistic
                            p1, p2, p3, p4 = param_set
                            distorted = func(img, p1, p2, p3, p4)
                        param_str = name_template.format(i)  # уровень
                        out_filename = f"{idx}_{param_str}.png"
                        out_path = os.path.join(distorted_folder, out_filename)
                        cv2.imwrite(out_path, distorted)
                        pbar.update(1)
                else:
                    # Для функций с одним числовым параметром
                    for param in params:
                        distorted = func(img, param)
                        if isinstance(param, float):
                            param_str = str(param).replace('.', '_')
                        else:
                            param_str = str(param)
                        out_filename = f"{idx}_{name_template.format(param_str)}.png"
                        out_path = os.path.join(distorted_folder, out_filename)
                        cv2.imwrite(out_path, distorted)
                        pbar.update(1)

    print(f"\nГотово! Искажённые изображения сохранены в: {distorted_folder}")

# -------------------- Точка входа --------------------
if __name__ == "__main__":
    # Задайте свои пути
    INPUT_DIR = "../dataset"            # папка с исходными PNG (1.png, 2.png, ...)
    DISTORTED_DIR = "../../dataset/photo"  # папка для результатов

    # Укажите номер, с которого начать (если нужно прервать и продолжить)
    START_FROM = 429

    process_all(INPUT_DIR, DISTORTED_DIR, start_from=START_FROM)



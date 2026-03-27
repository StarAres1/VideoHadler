import json
import cv2
import numpy as np
from app.core.ContrastImprover import ContrastImprover
import piq
import argparse
from tqdm import tqdm
import os
import re
import time

import torch
import torch_directml

# Определяем устройство DirectML (автоматически выберет лучший GPU)
device = torch_directml.device()
print(f"Используется устройство: {device}")

# В функции img_to_tensor оставляем тензор на CPU, затем перемещаем
def img_to_tensor(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
    tensor = tensor.unsqueeze(0) / 255.0
    return tensor.to(device)   # перемещаем на GPU через DirectML

# ---------------------- Вспомогательные функции ----------------------

def compute_rms_contrast(img):
    """Среднеквадратичный контраст (стандартное отклонение яркости)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    return float(gray.std())

def compute_michelson_contrast(img):
    """Контраст Микельсона (max-min)/(max+min)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    min_val = gray.min()
    max_val = gray.max()
    if max_val + min_val == 0:
        return 0.0
    return float((max_val - min_val) / (max_val + min_val))

def compute_laplacian_variance(img):
    """Дисперсия лапласиана (мера резкости)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())

def compute_hist_correlation(img, ref):
    """Корреляция гистограмм яркостей между img и ref"""
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    hist_img = cv2.calcHist([gray_img], [0], None, [256], [0,256])
    hist_ref = cv2.calcHist([gray_ref], [0], None, [256], [0,256])
    cv2.normalize(hist_img, hist_img)
    cv2.normalize(hist_ref, hist_ref)
    return float(cv2.compareHist(hist_img, hist_ref, cv2.HISTCMP_CORREL))

# ---------------------- Генерация методов улучшения ----------------------
def generate_enhanced_images(low_img):
    """
    Генерирует все улучшенные версии low_img с помощью предопределённых методов.
    Возвращает список кортежей: (изображение, название_метода, словарь_параметров)
    """
    specs = []

    # CLAHE (оставлено 6 вариаций)
    clahe_params = [
        (1.0, 4, 4),
        (1.5, 4, 4),
        (2.0, 4, 4),
        (2.5, 4, 4),
        (3.0, 4, 4),
        (4.0, 4, 4)
    ]
    for clip, gx, gy in clahe_params:
        img = ContrastImprover.CLAHE(low_img, clipLimit=clip, titleGridSizeX=gx, titleGridSizeY=gy)
        specs.append((img, "CLAHE", {"clipLimit": clip, "gridX": gx, "gridY": gy}))

    # Гамма-коррекция (13 уровней)
    gamma_values = [0.4, 0.48, 0.55, 0.63, 0.71, 0.86, 1.17, 1.33, 1.48, 1.64, 1.79, 1.95, 2.1]
    for gamma in gamma_values:
        img = ContrastImprover.gamma_correction(low_img, gamma=gamma)
        specs.append((img, "gamma", {"gamma": round(gamma, 2)}))

    # Гистограммная эквализация
    img_he = ContrastImprover.HE(low_img)
    specs.append((img_he, "HE", {}))

    return specs

# ---------------------- Обработка одной пары с таймерами ----------------------
def process_image_pair(low_path, ref_path):
    """
    Для заданного искажённого изображения и эталона находит лучший метод улучшения.
    Возвращает словарь с названием метода, параметрами (времена больше не возвращаются).
    """
    times = {}

    # ---- Открытие low изображения ----
    t_start = time.perf_counter()
    low_img = cv2.imread(low_path)
    times['open_low'] = time.perf_counter() - t_start
    if low_img is None:
        print(f"Предупреждение: не удалось прочитать {low_path}")
        return None

    # ---- Открытие ref изображения ----
    t_start = time.perf_counter()
    ref_img = cv2.imread(ref_path)
    times['open_ref'] = time.perf_counter() - t_start
    if ref_img is None:
        print(f"Предупреждение: не удалось прочитать {ref_path}")
        return None

    # ---- Генерация улучшенных версий ----
    t_start = time.perf_counter()
    enhanced_list = generate_enhanced_images(low_img)
    times['generate_enhanced'] = time.perf_counter() - t_start
    num = len(enhanced_list)

    # ---- Подсчет SSIM (включая преобразование в тензоры) ----
    t_start = time.perf_counter()
    ref_tensor = img_to_tensor(ref_img)
    tensors = [img_to_tensor(img) for img, _, _ in enhanced_list]
    ssim_vals = [float(piq.ssim(ref_tensor, ten, data_range=1.0)) for ten in tensors]
    times['compute_ssim'] = time.perf_counter() - t_start

    # ---- Подсчет контрастных метрик (rms, michelson, laplacian, hist_corr) ----
    t_start = time.perf_counter()
    results = []
    for idx, (img, method, params) in enumerate(enhanced_list):
        rms = compute_rms_contrast(img)
        mic = compute_michelson_contrast(img)
        lap = compute_laplacian_variance(img)
        hist_corr = compute_hist_correlation(img, ref_img)
        results.append({
            "index": idx,
            "method": method,
            "params": params,
            "ssim": ssim_vals[idx],
            "rms": rms,
            "michelson": mic,
            "laplacian": lap,
            "hist_corr": hist_corr
        })
    times['compute_contrast_metrics'] = time.perf_counter() - t_start

    # ---- Метрики эталонного изображения ----
    ref_rms = compute_rms_contrast(ref_img)
    ref_michelson = compute_michelson_contrast(ref_img)
    ref_laplacian = compute_laplacian_variance(ref_img)

    # Фиксированные сигмы для RMS и Микельсона (диапазон 0-1)
    sigma_rms = 0.1
    sigma_michelson = 0.1

    # Сигма для лапласиана (вычисляем по всем улучшенным)
    lap_vals = [r["laplacian"] for r in results]
    sigma_laplacian = np.std(lap_vals) if len(lap_vals) > 1 else 1.0

    # Построение оценок (чем выше, тем лучше)
    scores = {key: [] for key in ["ssim", "hist_corr", "rms_score", "michelson_score", "laplacian_score"]}
    for r in results:
        scores["ssim"].append(r["ssim"])
        scores["hist_corr"].append(r["hist_corr"])

        # Гауссово расстояние до эталона
        rms_diff = r["rms"] - ref_rms
        scores["rms_score"].append(np.exp(- (rms_diff**2) / (2 * sigma_rms**2)))

        mic_diff = r["michelson"] - ref_michelson
        scores["michelson_score"].append(np.exp(- (mic_diff**2) / (2 * sigma_michelson**2)))

        lap_diff = r["laplacian"] - ref_laplacian
        scores["laplacian_score"].append(np.exp(- (lap_diff**2) / (2 * sigma_laplacian**2)))

    # ---- Подсчет итогового рейтинга и выбор лучшего метода ----
    t_start = time.perf_counter()
    # Ранжирование по каждой метрике (1 — лучший)
    ranks = {metric: [0]*num for metric in ["ssim", "hist_corr", "rms", "michelson", "laplacian"]}
    for metric, score_list in scores.items():
        sorted_idx = sorted(range(num), key=lambda i: score_list[i], reverse=True)
        for rank, idx in enumerate(sorted_idx, start=1):
            if metric == "rms_score":
                ranks["rms"][idx] = rank
            elif metric == "michelson_score":
                ranks["michelson"][idx] = rank
            elif metric == "laplacian_score":
                ranks["laplacian"][idx] = rank
            else:
                ranks[metric][idx] = rank

    # Средний ранг по пяти метрикам
    avg_ranks = []
    for i in range(num):
        avg = np.mean([ranks["ssim"][i], ranks["hist_corr"][i],
                       ranks["rms"][i], ranks["michelson"][i], ranks["laplacian"][i]])
        avg_ranks.append(avg)

    # Выбор лучшего (наименьший средний ранг)
    best_idx = int(np.argmin(avg_ranks))
    best = results[best_idx]
    times['compute_ranking'] = time.perf_counter() - t_start

    # Возвращаем только метод и параметры (времена не включаем)
    return {
        "method": best["method"],
        "params": best["params"]
    }

# ---------------------- Загрузка уже обработанных ключей ----------------------
def load_processed_keys(output_file):
    """
    Если файл существует, читает все строки, парсит JSON и возвращает множество
    значений из поля "image". Иначе возвращает пустое множество.
    """
    if not os.path.isfile(output_file):
        return set()
    processed = set()
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "image" in data:
                    processed.add(data["image"])
            except json.JSONDecodeError:
                continue
    return processed

# ---------------------- Основной скрипт ----------------------
def main():
    parser = argparse.ArgumentParser(description="Выбор наилучшего метода улучшения контраста для каждого искажённого изображения (с дозаписью и сортировкой).")
    parser.add_argument("--low_dir", type=str, default="../../dataset/half", help="Папка с искажёнными изображениями")
    parser.add_argument("--ref_dir", type=str, default="../half_ref", help="Папка с эталонными (оригинальными) изображениями")
    parser.add_argument("--output", type=str, default="format_ssim_contrasts_20_2.jsonl", help="Выходной файл в формате JSON Lines (по умолчанию best_methods.jsonl)")
    parser.add_argument("--ext", type=str, default=".png", help="Расширение файлов (по умолчанию .png)")
    parser.add_argument("--start_num", type=int, default=711, help="Начинать обработку с изображений, чей номер референса не меньше указанного (по умолчанию 1)")
    parser.add_argument("--end_num", type=int, default=None, help="Заканчивать обработку на изображении с указанным номером (включительно). Если не задано, обрабатываются все от start_num и до конца.")
    args = parser.parse_args()

    # Получаем список искажённых файлов
    all_files = [f for f in os.listdir(args.low_dir) if f.lower().endswith(args.ext)]
    if not all_files:
        print(f"Файлы с расширением {args.ext} не найдены в {args.low_dir}")
        return

    # Извлекаем номер референса из каждого имени
    file_data = []  # (числовой_номер, имя_файла)
    pattern = re.compile(r"^(\d+)_.*")
    for f in all_files:
        match = pattern.match(f)
        if match:
            num = int(match.group(1))
            file_data.append((num, f))
        else:
            print(f"Предупреждение: имя файла {f} не соответствует формату 'число_...', пропускаем.")

    if not file_data:
        print("Нет файлов, соответствующих шаблону.")
        return

    # Сортируем по числовому номеру
    file_data.sort(key=lambda x: x[0])

    # Фильтруем по диапазону номеров
    filtered_data = []
    for num, f in file_data:
        if num >= args.start_num:
            if args.end_num is not None and num > args.end_num:
                continue
            filtered_data.append((num, f))

    if not filtered_data:
        print(f"Нет изображений с номерами от {args.start_num} до {args.end_num if args.end_num else '∞'}.")
        return

    # Загружаем уже обработанные ключи из выходного файла
    processed_keys = load_processed_keys(args.output)

    # Открываем выходной файл в режиме добавления
    with open(args.output, 'a', encoding='utf-8') as f_out:
        for num, low_file in tqdm(filtered_data, desc="Обработка изображений"):
            key = os.path.splitext(low_file)[0]
            if key in processed_keys:
                continue

            ref_file = f"{num}{args.ext}"
            low_path = os.path.join(args.low_dir, low_file)
            ref_path = os.path.join(args.ref_dir, ref_file)

            if not os.path.isfile(ref_path):
                print(f"Предупреждение: эталон {ref_path} не найден, пропускаем {low_file}")
                continue

            result = process_image_pair(low_path, ref_path)
            if result is not None:
                output_record = {
                    "image": key,
                    "method": result["method"],
                    "params": result["params"]
                }
                f_out.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                f_out.flush()
                processed_keys.add(key)

    print(f"\nРезультаты сохранены/добавлены в {args.output} (формат JSON Lines)")

if __name__ == "__main__":
    main()
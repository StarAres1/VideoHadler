import json
import cv2
import numpy as np
from app.core.ContrastImprover import ContrastImprover
import piq
import argparse
from tqdm import tqdm
import os
import re

import torch
import torch_directml

device = torch_directml.device()
print(f"Используется устройство: {device}")

def img_to_tensor(img):
    """Преобразование BGR -> RGB -> тензор на GPU"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
    tensor = tensor.unsqueeze(0) / 255.0
    return tensor.to(device)

def compute_rms_contrast(img):
    """Глобальный среднеквадратичный контраст (стандартное отклонение яркости)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    return float(gray.std())

def generate_13_methods(low_img):
    """Генерирует ровно 13 методов улучшения (как в первом коде)"""
    specs = []

    # 1–2. CLAHE
    clahe_params = [(1.0, 4, 4), (3.0, 4, 4)]
    for clip, gx, gy in clahe_params:
        img = ContrastImprover.CLAHE(low_img, clipLimit=clip, titleGridSizeX=gx, titleGridSizeY=gy)
        specs.append((img, "CLAHE", {"clipLimit": clip, "gridX": gx, "gridY": gy}))

    # 3–4. adjust_contrast (Retinex)
    adjust_params = [(2.0, 8), (2.5, 10)]
    for alpha, beta in adjust_params:
        img = ContrastImprover.adjust_contrast(low_img, alpha=alpha, beta=beta)
        specs.append((img, "adjust_contrast", {"alpha": alpha, "beta": beta}))

    # 5–9. gamma_correction (5 значений)
    gamma_values = np.linspace(0.5, 1.9, 5)  # 0.5, 0.85, 1.2, 1.55, 1.9
    for gamma in gamma_values:
        img = ContrastImprover.gamma_correction(low_img, gamma=gamma)
        specs.append((img, "gamma", {"gamma": round(gamma, 2)}))

    # 10–12. sigmoid_correction (3 варианта)
    sigmoid_params = [(0.3, 10), (0.3, 12), (0.4, 12)]
    for cutoff, gain in sigmoid_params:
        img = ContrastImprover.sigmoid_correction(low_img, cutoff=cutoff, gain=gain)
        specs.append((img, "sigmoid", {"cutoff": cutoff, "gain": gain}))

    # 13. sigmoid + HE
    img_sigmoid = ContrastImprover.sigmoid_correction(low_img, cutoff=0.3, gain=12)
    img_combined = ContrastImprover.HE(img_sigmoid)
    specs.append((img_combined, "sigmoid+HE", {"cutoff": 0.3, "gain": 12}))

    return specs

def process_image_pair(low_path, ref_path, w_ssim=0.3, w_rms=0.7):
    """
    Выбирает лучший метод, максимизируя взвешенную сумму нормализованных значений
    SSIM (сравнение с референсом) и глобального RMS-контраста.
    """
    low_img = cv2.imread(low_path)
    if low_img is None:
        print(f"Не удалось прочитать {low_path}")
        return None
    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        print(f"Не удалось прочитать {ref_path}")
        return None

    enhanced_list = generate_13_methods(low_img)
    num = len(enhanced_list)  # должно быть 13

    # ---- SSIM (относительно референса) ----
    ref_tensor = img_to_tensor(ref_img)
    tensors = [img_to_tensor(img) for img, _, _ in enhanced_list]
    ssim_vals = [float(piq.ssim(ref_tensor, ten, data_range=1.0)) for ten in tensors]

    # ---- RMS-контраст (глобальный, чем выше, тем лучше) ----
    rms_vals = [compute_rms_contrast(img) for img, _, _ in enhanced_list]

    # ---- Нормализация в плавающем окне (min-max) ----
    min_ssim, max_ssim = min(ssim_vals), max(ssim_vals)
    min_rms, max_rms = min(rms_vals), max(rms_vals)

    # Избегаем деления на ноль
    if max_ssim > min_ssim:
        norm_ssim = [(s - min_ssim) / (max_ssim - min_ssim) for s in ssim_vals]
    else:
        norm_ssim = [0.0] * num

    if max_rms > min_rms:
        norm_rms = [(r - min_rms) / (max_rms - min_rms) for r in rms_vals]
    else:
        norm_rms = [0.0] * num

    # ---- Взвешенная сумма ----
    combined_scores = [w_ssim * ns + w_rms * nr for ns, nr in zip(norm_ssim, norm_rms)]
    best_idx = int(np.argmax(combined_scores))
    best_method, best_params = enhanced_list[best_idx][1], enhanced_list[best_idx][2]

    return {
        "method": best_method,
        "params": best_params
    }

def load_processed_keys(output_file):
    """Загружает уже обработанные ключи из JSONL"""
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

def main():
    parser = argparse.ArgumentParser(
        description="Выбор лучшего метода улучшения контраста по SSIM (сравнение с эталоном) и глобальному RMS (максимизация)."
    )
    parser.add_argument("--low_dir", type=str, default="../../dataset/half")
    parser.add_argument("--ref_dir", type=str, default="../half_ref")
    parser.add_argument("--output", type=str, default="best_methods(huge_ssim)2.jsonl")
    parser.add_argument("--ext", type=str, default=".png")
    parser.add_argument("--start_num", type=int, default=856)
    parser.add_argument("--end_num", type=int, default=None)
    parser.add_argument("--w_ssim", type=float, default=0.6, help="Вес SSIM (0..1)")
    parser.add_argument("--w_rms", type=float, default=0.4, help="Вес RMS-контраста (0..1)")
    args = parser.parse_args()

    # Проверка весов
    if abs(args.w_ssim + args.w_rms - 1.0) > 1e-6:
        print("Внимание: сумма весов не равна 1, будет использовано соотношение без нормировки.")

    # Сбор файлов
    all_files = [f for f in os.listdir(args.low_dir) if f.lower().endswith(args.ext)]
    if not all_files:
        print(f"Файлы {args.ext} не найдены в {args.low_dir}")
        return

    pattern = re.compile(r"^(\d+)_.*")
    file_data = []
    for f in all_files:
        m = pattern.match(f)
        if m:
            file_data.append((int(m.group(1)), f))
        else:
            print(f"Предупреждение: {f} не соответствует шаблону, пропущен.")

    file_data.sort(key=lambda x: x[0])
    filtered_data = [(num, f) for num, f in file_data
                     if num >= args.start_num and (args.end_num is None or num <= args.end_num)]

    if not filtered_data:
        print(f"Нет изображений в диапазоне {args.start_num}–{args.end_num if args.end_num else '∞'}.")
        return

    processed_keys = load_processed_keys(args.output)

    with open(args.output, 'a', encoding='utf-8') as f_out:
        for num, low_file in tqdm(filtered_data, desc="Обработка"):
            key = os.path.splitext(low_file)[0]
            if key in processed_keys:
                continue

            ref_file = f"{num}{args.ext}"
            low_path = os.path.join(args.low_dir, low_file)
            ref_path = os.path.join(args.ref_dir, ref_file)

            if not os.path.isfile(ref_path):
                print(f"Эталон {ref_path} не найден, пропускаем {low_file}")
                continue

            result = process_image_pair(low_path, ref_path,
                                        w_ssim=args.w_ssim,
                                        w_rms=args.w_rms)
            if result is not None:
                record = {"image": key, "method": result["method"], "params": result["params"]}
                f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
                f_out.flush()
                processed_keys.add(key)

    print(f"\nРезультаты сохранены в {args.output}")

if __name__ == "__main__":
    main()
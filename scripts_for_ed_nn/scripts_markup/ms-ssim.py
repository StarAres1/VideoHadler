import json
import cv2
import torch
import numpy as np
from app.core.ContrastImprover import ContrastImprover
from pytorch_msssim import ms_ssim
import time

path_file = "../../dataset/train/format.json"
path_photo = "1_contrast_change_alpha_0_5.png"
path_ref = "1.png"


def img_to_tensor(img):
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert to tensor (H, W, C) -> (C, H, W)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
    # Add batch dimension (N, C, H, W)
    tensor = tensor.unsqueeze(0)
    # Normalize to [0, 1]
    tensor = tensor / 255.0
    return tensor


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


with open(path_file, "w", encoding="utf-8") as file:
    data = {}
    photo = cv2.imread(path_photo)
    ref = cv2.imread(path_ref)
    ref_tensor = img_to_tensor(ref)

    # Вывод метрик для эталонного и искажённого изображений
    print("Метрики для эталонного изображения (ref):")
    print(f"  RMS contrast: {compute_rms_contrast(ref):.4f}")
    print(f"  Michelson contrast: {compute_michelson_contrast(ref):.4f}")
    print(f"  Laplacian variance: {compute_laplacian_variance(ref):.4f}")

    print("\nМетрики для искажённого изображения (photo):")
    print(f"  RMS contrast: {compute_rms_contrast(photo):.4f}")
    print(f"  Michelson contrast: {compute_michelson_contrast(photo):.4f}")
    print(f"  Laplacian variance: {compute_laplacian_variance(photo):.4f}\n")

    start1 = time.time()

    # --- CLAHE с новыми параметрами ---
    photo1 = ContrastImprover.CLAHE(photo, clipLimit=1.0, titleGridSizeX=4, titleGridSizeY=4)
    photo2 = ContrastImprover.CLAHE(photo, clipLimit=1.5, titleGridSizeX=4, titleGridSizeY=4)
    photo3 = ContrastImprover.CLAHE(photo, clipLimit=2.0, titleGridSizeX=4, titleGridSizeY=4)
    photo4 = ContrastImprover.CLAHE(photo, clipLimit=2.5, titleGridSizeX=4, titleGridSizeY=4)
    photo5 = ContrastImprover.CLAHE(photo, clipLimit=3.0, titleGridSizeX=4, titleGridSizeY=4)
    photo6 = ContrastImprover.CLAHE(photo, clipLimit=2.0, titleGridSizeX=8, titleGridSizeY=8)

    # --- Retinex с новыми параметрами ---
    photo7 = ContrastImprover.adjust_contrast(photo, alpha=2.0, beta=8)
    photo8 = ContrastImprover.adjust_contrast(photo, alpha=2.0, beta=10)
    photo9 = ContrastImprover.adjust_contrast(photo, alpha=2.0, beta=12)
    photo10 = ContrastImprover.adjust_contrast(photo, alpha=1.8, beta=9)
    photo11 = ContrastImprover.adjust_contrast(photo, alpha=1.8, beta=11)
    photo12 = ContrastImprover.adjust_contrast(photo, alpha=2.2, beta=9)
    photo13 = ContrastImprover.adjust_contrast(photo, alpha=2.2, beta=11)
    photo14 = ContrastImprover.adjust_contrast(photo, alpha=2.5, beta=10)

    # --- Гамма-коррекция с умеренными параметрами ---
    photo15 = ContrastImprover.gamma_correction(photo, gamma=0.8)
    photo16 = ContrastImprover.gamma_correction(photo, gamma=0.9)
    photo17 = ContrastImprover.gamma_correction(photo, gamma=1.0)
    photo18 = ContrastImprover.gamma_correction(photo, gamma=1.1)
    photo19 = ContrastImprover.gamma_correction(photo, gamma=1.2)
    photo20 = ContrastImprover.gamma_correction(photo, gamma=1.5)

    # --- Сигмоидальная коррекция с небольшим усилением ---
    photo21 = ContrastImprover.sigmoid_correction(photo, cutoff=0.5, gain=2)
    photo22 = ContrastImprover.sigmoid_correction(photo, cutoff=0.5, gain=3)
    photo23 = ContrastImprover.sigmoid_correction(photo, cutoff=0.4, gain=3)
    photo24 = ContrastImprover.sigmoid_correction(photo, cutoff=0.6, gain=3)
    photo25 = ContrastImprover.sigmoid_correction(photo, cutoff=0.5, gain=4)

    # --- Оригинал (без обработки) ---
    photo_original = photo.copy()
    photo26 = photo_original

    # --- Остальные методы ---
    photo27 = ContrastImprover.HE(photo)
    photo28 = ContrastImprover.auto_gamma(photo)
    photo29 = ContrastImprover.combined_enhancement(photo)

    start2 = time.time()
    print(f"Время обработки: {start2 - start1:.3f} сек")

    # Сохранение результатов
    for i in range(1, 30):
        cv2.imwrite(f"photo{i}.png", locals()[f"photo{i}"])

    start3 = time.time()
    print(f"Сохранение заняло: {start3 - start2:.3f} сек")

    # Вычисление MS-SSIM (без ускорений, просто цикл)
    start_ms = time.time()
    data_mssim = {}
    for i in range(1, 30):
        key = f"photo{i}"
        img = locals()[key]
        data_mssim[key] = float(ms_ssim(ref_tensor, img_to_tensor(img)))
    ms_time = time.time() - start_ms
    print(f"Вычисление MS-SSIM заняло: {ms_time:.3f} сек")

    # Словарь с описанием методов для каждого photoN
    method_desc = {
        "photo1": "CLAHE clipLimit=1.0 grid=4x4",
        "photo2": "CLAHE clipLimit=1.5 grid=4x4",
        "photo3": "CLAHE clipLimit=2.0 grid=4x4",
        "photo4": "CLAHE clipLimit=2.5 grid=4x4",
        "photo5": "CLAHE clipLimit=3.0 grid=4x4",
        "photo6": "CLAHE clipLimit=2.0 grid=8x8",
        "photo7": "Retinex alpha=2.0 beta=8",
        "photo8": "Retinex alpha=2.0 beta=10",
        "photo9": "Retinex alpha=2.0 beta=12",
        "photo10": "Retinex alpha=1.8 beta=9",
        "photo11": "Retinex alpha=1.8 beta=11",
        "photo12": "Retinex alpha=2.2 beta=9",
        "photo13": "Retinex alpha=2.2 beta=11",
        "photo14": "Retinex alpha=2.5 beta=10",
        "photo15": "gamma_correction gamma=0.8",
        "photo16": "gamma_correction gamma=0.9",
        "photo17": "gamma_correction gamma=1.0 (original)",
        "photo18": "gamma_correction gamma=1.1",
        "photo19": "gamma_correction gamma=1.2",
        "photo20": "gamma_correction gamma=1.5",
        "photo21": "sigmoid_correction cutoff=0.5 gain=2",
        "photo22": "sigmoid_correction cutoff=0.5 gain=3",
        "photo23": "sigmoid_correction cutoff=0.4 gain=3",
        "photo24": "sigmoid_correction cutoff=0.6 gain=3",
        "photo25": "sigmoid_correction cutoff=0.5 gain=4",
        "photo26": "original (no enhancement)",
        "photo27": "HE",
        "photo28": "auto_gamma",
        "photo29": "combined_enhancement",
    }

    # Сбор всех результатов с дополнительными метриками
    results = []
    for i in range(1, 30):
        key = f"photo{i}"
        img = locals()[key]
        desc = method_desc[key]
        mssim = data_mssim[key]
        rms = compute_rms_contrast(img)
        mic = compute_michelson_contrast(img)
        lap = compute_laplacian_variance(img)
        results.append({
            "key": key,
            "desc": desc,
            "ms_ssim": mssim,
            "rms_contrast": rms,
            "michelson": mic,
            "laplacian_var": lap
        })

    # Сортировка по убыванию MS-SSIM
    results.sort(key=lambda x: x["ms_ssim"], reverse=True)

    # Вывод таблицы
    print("\n=== Рейтинг изображений по MS-SSIM (от лучшего к худшему) ===")
    header = f"{'Key':<8} {'MS-SSIM':<8} {'RMS':<8} {'Michelson':<10} {'Laplacian':<10} {'Description'}"
    print(header)
    for r in results:
        print(f"{r['key']:<8} {r['ms_ssim']:.4f}   {r['rms_contrast']:.4f}   {r['michelson']:.4f}   {r['laplacian_var']:.2f}   {r['desc']}")

    # Сохранение всех данных в JSON (можно расширить)
    json.dump(data_mssim, file, indent=4)
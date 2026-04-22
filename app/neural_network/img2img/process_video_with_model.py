from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import pickle

try:
    import torch_directml
except ImportError:
    torch_directml = None


# ---------------------------
# Настройки (можно менять)
# ---------------------------
MODEL_PATH = Path(r"C:\Projects\QtTrial\app\neural_network\models\img2img\image2image_runs\best_model.pth")
VIDEO_PATH = Path(r"C:\Users\Андрей\Videos\Тестовые видео\1.mp4")
TARGET_WIDTH = 700
TARGET_HEIGHT = 400
TARGET_SIZE_HW = (TARGET_HEIGHT, TARGET_WIDTH)

# С какой минуты видео начать обработку
START_MINUTE = 0.0
# Максимальное число обрабатываемых кадров (None = без ограничения)
MAX_FRAMES = 700
# Пути выхода (None -> автоматически рядом с входным видео)
OUTPUT_VIDEO_PATH = None
REPORT_PATH = None


class ResUNetEnhancer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=3,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


def get_device():
    if torch_directml is not None:
        return torch_directml.device()
    return torch.device("cpu")


def to_landscape_rgb(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.shape[0] > image_rgb.shape[1]:
        return np.rot90(image_rgb, k=3).copy()
    return image_rgb


def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = to_landscape_rgb(rgb)
    resized = cv2.resize(rgb, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)
    return resized.astype(np.float32) / 255.0


def postprocess_to_bgr(pred_rgb_01: np.ndarray) -> np.ndarray:
    pred_rgb_8u = np.clip(pred_rgb_01 * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(pred_rgb_8u, cv2.COLOR_RGB2BGR)


def rms_contrast_rgb01(image_rgb_01: np.ndarray) -> float:
    gray = 0.299 * image_rgb_01[..., 0] + 0.587 * image_rgb_01[..., 1] + 0.114 * image_rgb_01[..., 2]
    mean_val = float(np.mean(gray))
    return float(np.sqrt(np.mean((gray - mean_val) ** 2)))


def build_output_paths(video_path: Path):
    out_video = video_path.with_name(video_path.stem + "_processed.mp4")
    out_report = video_path.with_name(video_path.stem + "_rms_report.txt")
    return out_video, out_report


def write_report(report_path: Path, rows: list, avg_rms_before: float, avg_rms_after: float, avg_gain_pct: float) -> None:
    with report_path.open("w", encoding="utf-8") as f:
        f.write("ОТЧЕТ ПО ОБРАБОТКЕ ВИДЕО НЕЙРОСЕТЬЮ\n")
        f.write("=" * 80 + "\n")
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Модель: {MODEL_PATH}\n")
        f.write(f"Размер обработки: {TARGET_WIDTH}x{TARGET_HEIGHT}, RGB\n")
        f.write(f"START_MINUTE: {START_MINUTE}\n")
        f.write(f"MAX_FRAMES: {MAX_FRAMES}\n")
        f.write("-" * 80 + "\n\n")

        for row in rows:
            f.write(
                f"frame={row['frame_index']}, time_sec={row['time_sec']:.3f}, "
                f"rms_before={row['rms_before']:.6f}, rms_after={row['rms_after']:.6f}, "
                f"gain_pct={row['gain_pct']:.2f}\n"
            )

        f.write("\n" + "-" * 80 + "\n")
        f.write("Средние значения:\n")
        f.write(f"avg_rms_before={avg_rms_before:.6f}\n")
        f.write(f"avg_rms_after={avg_rms_after:.6f}\n")
        f.write(f"avg_gain_pct={avg_gain_pct:.2f}\n")


def main() -> None:
    video_path = VIDEO_PATH
    if not video_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Файл модели не найден: {MODEL_PATH}")

    default_out_video, default_report = build_output_paths(video_path)
    out_video_path = OUTPUT_VIDEO_PATH if OUTPUT_VIDEO_PATH is not None else default_out_video
    report_path = REPORT_PATH if REPORT_PATH is not None else default_report

    device = get_device()
    print(f"Device: {device}")
    if torch_directml is None:
        print("torch_directml не установлен, используется CPU.")

    model = ResUNetEnhancer().to(device)
    # Сначала пробуем строгую безопасную загрузку только весов.
    # Если файл сохранен в другом формате, используем fallback для доверенного локального файла.
    try:
        loaded_obj = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        print("weights_only=True не сработал, выполняем fallback weights_only=False для локальной модели.")
        loaded_obj = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    if isinstance(loaded_obj, dict) and "model_state_dict" in loaded_obj:
        state_dict = loaded_obj["model_state_dict"]
    else:
        state_dict = loaded_obj

    model.load_state_dict(state_dict)
    model.eval()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    start_frame = int(START_MINUTE * 60.0 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (TARGET_WIDTH, TARGET_HEIGHT))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Не удалось создать видео для записи: {out_video_path}")

    rows = []
    processed = 0
    current_frame_idx = start_frame

    with torch.no_grad():
        while True:
            if MAX_FRAMES is not None and processed >= MAX_FRAMES:
                break

            ok, frame_bgr = cap.read()
            if not ok:
                break

            input_rgb_01 = preprocess_frame(frame_bgr)
            input_tensor = torch.from_numpy(input_rgb_01).permute(2, 0, 1).unsqueeze(0).to(device)
            output = model(input_tensor)
            pred_chw = output[0].detach().cpu().numpy()
            pred_rgb_01 = np.transpose(pred_chw, (1, 2, 0))

            out_bgr = postprocess_to_bgr(pred_rgb_01)
            writer.write(out_bgr)

            rms_before = rms_contrast_rgb01(input_rgb_01)
            rms_after = rms_contrast_rgb01(pred_rgb_01)
            gain_pct = ((rms_after - rms_before) / (rms_before + 1e-12)) * 100.0

            rows.append(
                {
                    "frame_index": current_frame_idx,
                    "time_sec": current_frame_idx / fps,
                    "rms_before": rms_before,
                    "rms_after": rms_after,
                    "gain_pct": gain_pct,
                }
            )

            processed += 1
            current_frame_idx += 1
            if processed % 25 == 0:
                print(f"Обработано кадров: {processed}")

    cap.release()
    writer.release()

    if not rows:
        raise RuntimeError("Не обработано ни одного кадра. Проверь START_MINUTE и входное видео.")

    avg_rms_before = float(np.mean([r["rms_before"] for r in rows]))
    avg_rms_after = float(np.mean([r["rms_after"] for r in rows]))
    avg_gain_pct = float(np.mean([r["gain_pct"] for r in rows]))
    write_report(report_path, rows, avg_rms_before, avg_rms_after, avg_gain_pct)

    print(f"Готово. Выходное видео: {out_video_path}")
    print(f"Отчет: {report_path}")


if __name__ == "__main__":
    main()

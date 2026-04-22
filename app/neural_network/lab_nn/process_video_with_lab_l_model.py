from datetime import datetime
from pathlib import Path
import pickle

import cv2
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

try:
    import torch_directml
except ImportError:
    torch_directml = None

try:
    import pyiqa
except ImportError:
    pyiqa = None


# ---------------------------
# Настройки (запуск из PyCharm)
# ---------------------------
MODEL_PATH = Path(r"C:\Projects\QtTrial\app\neural_network\lab_nn\lab_l_runs\best_model.pth")
VIDEO_PATH = Path(r"C:\Users\Андрей\Videos\Тестовые видео\11.mp4")

OUTPUT_VIDEO_PATH = None   # None => рядом с исходным видео
REPORT_PATH = None         # None => рядом с исходным видео

TARGET_WIDTH = 700
TARGET_HEIGHT = 400
TARGET_SIZE = (TARGET_WIDTH, TARGET_HEIGHT)

START_MINUTE = 0.0
MAX_FRAMES = 100           # None => до конца видео


class ResUNetEnhancerL(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


class NoReferenceIqaEvaluator:
    def __init__(self) -> None:
        self.available = False
        self.metric_brisque = None
        self.metric_niqe = None
        self.metric_piqe = None

        if pyiqa is None:
            return
        try:
            self.metric_brisque = pyiqa.create_metric("brisque", device="cpu")
            self.metric_niqe = pyiqa.create_metric("niqe", device="cpu")
            self.metric_piqe = pyiqa.create_metric("piqe", device="cpu")
            self.available = True
        except Exception:
            self.available = False

    def evaluate_l(self, image_hw: np.ndarray):
        if not self.available:
            nan = float("nan")
            return nan, nan, nan

        tensor = torch.from_numpy(image_hw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        tensor_rgb = tensor.repeat(1, 3, 1, 1)
        with torch.no_grad():
            brisque = float(self.metric_brisque(tensor_rgb).item())
            niqe = float(self.metric_niqe(tensor_rgb).item())
            piqe = float(self.metric_piqe(tensor_rgb).item())
        return brisque, niqe, piqe


def get_device():
    if torch_directml is not None:
        return torch_directml.device()
    return torch.device("cpu")


def rms_contrast_l(image_hw: np.ndarray) -> float:
    return float(np.sqrt(np.mean((image_hw - np.mean(image_hw)) ** 2)))


def laplacian_variance(image_hw: np.ndarray) -> float:
    image_u8 = np.clip(image_hw * 255.0, 0, 255).astype(np.uint8)
    lap = cv2.Laplacian(image_u8, cv2.CV_64F)
    return float(lap.var())


def tenengrad_sharpness(image_hw: np.ndarray) -> float:
    image_u8 = np.clip(image_hw * 255.0, 0, 255).astype(np.uint8)
    gx = cv2.Sobel(image_u8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image_u8, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def image_entropy(image_hw: np.ndarray) -> float:
    image_u8 = np.clip(image_hw * 255.0, 0, 255).astype(np.uint8)
    hist = cv2.calcHist([image_u8], [0], None, [256], [0, 256]).ravel()
    p = hist / (np.sum(hist) + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def metric_bundle(image_l: np.ndarray, iqa: NoReferenceIqaEvaluator) -> dict:
    brisque, niqe, piqe = iqa.evaluate_l(image_l)
    return {
        "rms": rms_contrast_l(image_l),
        "lap_var": laplacian_variance(image_l),
        "tenengrad": tenengrad_sharpness(image_l),
        "entropy": image_entropy(image_l),
        "mean_l": float(np.mean(image_l)),
        "brisque": brisque,
        "niqe": niqe,
        "piqe": piqe,
    }


def pct_gain(before: float, after: float) -> float:
    return float(((after - before) / (abs(before) + 1e-12)) * 100.0)


def build_output_paths(video_path: Path):
    out_video = video_path.with_name(video_path.stem + "_lab_l_enhanced.mp4")
    out_report = video_path.with_name(video_path.stem + "_lab_l_report.txt")
    return out_video, out_report


def load_model(model_path: Path, device):
    model = ResUNetEnhancerL().to(device)
    try:
        loaded_obj = torch.load(model_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        loaded_obj = torch.load(model_path, map_location="cpu", weights_only=False)

    if isinstance(loaded_obj, dict) and "model_state_dict" in loaded_obj:
        state_dict = loaded_obj["model_state_dict"]
    else:
        state_dict = loaded_obj
    model.load_state_dict(state_dict)
    model.eval()
    return model


def write_report(report_path: Path, rows: list, avg: dict) -> None:
    with report_path.open("w", encoding="utf-8") as f:
        f.write("ОТЧЕТ ПО ОБРАБОТКЕ ВИДЕО (LAB L-channel enhancement)\n")
        f.write("=" * 100 + "\n")
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Модель: {MODEL_PATH}\n")
        f.write(f"START_MINUTE: {START_MINUTE}\n")
        f.write(f"MAX_FRAMES: {MAX_FRAMES}\n")
        f.write("-" * 100 + "\n\n")

        for r in rows:
            f.write(
                f"frame={r['frame_index']}, time_sec={r['time_sec']:.3f}, "
                f"RMS: {r['rms_before']:.6f}->{r['rms_after']:.6f} ({r['rms_gain_pct']:.2f}%), "
                f"LapVar: {r['lap_before']:.4f}->{r['lap_after']:.4f} ({r['lap_gain_pct']:.2f}%), "
                f"Tenengrad: {r['ten_before']:.4f}->{r['ten_after']:.4f} ({r['ten_gain_pct']:.2f}%), "
                f"Entropy: {r['ent_before']:.4f}->{r['ent_after']:.4f} ({r['ent_gain_pct']:.2f}%), "
                f"MeanL: {r['mean_before']:.4f}->{r['mean_after']:.4f} ({r['mean_gain_pct']:.2f}%), "
                f"BRISQUE: {r['brisque_before']:.4f}->{r['brisque_after']:.4f} ({r['brisque_gain_pct']:.2f}%), "
                f"NIQE: {r['niqe_before']:.4f}->{r['niqe_after']:.4f} ({r['niqe_gain_pct']:.2f}%), "
                f"PIQE: {r['piqe_before']:.4f}->{r['piqe_after']:.4f} ({r['piqe_gain_pct']:.2f}%)\n"
            )

        f.write("\n" + "-" * 100 + "\n")
        f.write("СРЕДНИЕ ПО ВСЕМ КАДРАМ\n")
        f.write(f"avg_rms_before={avg['rms_before']:.6f}, avg_rms_after={avg['rms_after']:.6f}, avg_rms_gain_pct={avg['rms_gain_pct']:.2f}\n")
        f.write(f"avg_lap_before={avg['lap_before']:.4f}, avg_lap_after={avg['lap_after']:.4f}, avg_lap_gain_pct={avg['lap_gain_pct']:.2f}\n")
        f.write(f"avg_ten_before={avg['ten_before']:.4f}, avg_ten_after={avg['ten_after']:.4f}, avg_ten_gain_pct={avg['ten_gain_pct']:.2f}\n")
        f.write(f"avg_ent_before={avg['ent_before']:.4f}, avg_ent_after={avg['ent_after']:.4f}, avg_ent_gain_pct={avg['ent_gain_pct']:.2f}\n")
        f.write(f"avg_mean_before={avg['mean_before']:.4f}, avg_mean_after={avg['mean_after']:.4f}, avg_mean_gain_pct={avg['mean_gain_pct']:.2f}\n")
        f.write(f"avg_brisque_before={avg['brisque_before']:.4f}, avg_brisque_after={avg['brisque_after']:.4f}, avg_brisque_gain_pct={avg['brisque_gain_pct']:.2f}\n")
        f.write(f"avg_niqe_before={avg['niqe_before']:.4f}, avg_niqe_after={avg['niqe_after']:.4f}, avg_niqe_gain_pct={avg['niqe_gain_pct']:.2f}\n")
        f.write(f"avg_piqe_before={avg['piqe_before']:.4f}, avg_piqe_after={avg['piqe_after']:.4f}, avg_piqe_gain_pct={avg['piqe_gain_pct']:.2f}\n")


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Видео не найдено: {VIDEO_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")

    out_video_default, report_default = build_output_paths(VIDEO_PATH)
    out_video = OUTPUT_VIDEO_PATH if OUTPUT_VIDEO_PATH is not None else out_video_default
    out_report = REPORT_PATH if REPORT_PATH is not None else report_default

    device = get_device()
    print(f"Device: {device}")
    if torch_directml is None:
        print("torch_directml не установлен, используется CPU.")

    model = load_model(MODEL_PATH, device)
    iqa = NoReferenceIqaEvaluator()
    if not iqa.available:
        print("pyiqa не найден/не инициализирован: BRISQUE/NIQE/PIQE будут nan.")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(START_MINUTE * 60.0 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Не удалось создать выходной файл: {out_video}")

    rows = []
    processed = 0
    frame_idx = start_frame

    with torch.no_grad():
        while True:
            if MAX_FRAMES is not None and processed >= MAX_FRAMES:
                break

            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
            l_orig = lab[..., 0].astype(np.float32) / 255.0

            l_for_model = cv2.resize(l_orig, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
            inp = torch.from_numpy(l_for_model).unsqueeze(0).unsqueeze(0).to(device)
            pred_small = model(inp)[0, 0].detach().cpu().numpy()
            pred_l = cv2.resize(pred_small, (frame_w, frame_h), interpolation=cv2.INTER_CUBIC)
            pred_l = np.clip(pred_l, 0.0, 1.0)

            before = metric_bundle(l_orig, iqa)
            after = metric_bundle(pred_l, iqa)

            lab_out = lab.copy()
            lab_out[..., 0] = np.clip(pred_l * 255.0, 0, 255).astype(np.uint8)
            rgb_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)
            bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
            writer.write(bgr_out)

            rows.append(
                {
                    "frame_index": frame_idx,
                    "time_sec": frame_idx / fps,
                    "rms_before": before["rms"],
                    "rms_after": after["rms"],
                    "rms_gain_pct": pct_gain(before["rms"], after["rms"]),
                    "lap_before": before["lap_var"],
                    "lap_after": after["lap_var"],
                    "lap_gain_pct": pct_gain(before["lap_var"], after["lap_var"]),
                    "ten_before": before["tenengrad"],
                    "ten_after": after["tenengrad"],
                    "ten_gain_pct": pct_gain(before["tenengrad"], after["tenengrad"]),
                    "ent_before": before["entropy"],
                    "ent_after": after["entropy"],
                    "ent_gain_pct": pct_gain(before["entropy"], after["entropy"]),
                    "mean_before": before["mean_l"],
                    "mean_after": after["mean_l"],
                    "mean_gain_pct": pct_gain(before["mean_l"], after["mean_l"]),
                    "brisque_before": before["brisque"],
                    "brisque_after": after["brisque"],
                    "brisque_gain_pct": pct_gain(before["brisque"], after["brisque"]),
                    "niqe_before": before["niqe"],
                    "niqe_after": after["niqe"],
                    "niqe_gain_pct": pct_gain(before["niqe"], after["niqe"]),
                    "piqe_before": before["piqe"],
                    "piqe_after": after["piqe"],
                    "piqe_gain_pct": pct_gain(before["piqe"], after["piqe"]),
                }
            )

            processed += 1
            frame_idx += 1
            if processed % 25 == 0:
                print(f"Обработано кадров: {processed}")

    cap.release()
    writer.release()

    if not rows:
        raise RuntimeError("Не обработано ни одного кадра. Проверь START_MINUTE/видео.")

    def _avg(key: str) -> float:
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        return float(np.nanmean(vals))

    avg = {
        "rms_before": _avg("rms_before"),
        "rms_after": _avg("rms_after"),
        "rms_gain_pct": _avg("rms_gain_pct"),
        "lap_before": _avg("lap_before"),
        "lap_after": _avg("lap_after"),
        "lap_gain_pct": _avg("lap_gain_pct"),
        "ten_before": _avg("ten_before"),
        "ten_after": _avg("ten_after"),
        "ten_gain_pct": _avg("ten_gain_pct"),
        "ent_before": _avg("ent_before"),
        "ent_after": _avg("ent_after"),
        "ent_gain_pct": _avg("ent_gain_pct"),
        "mean_before": _avg("mean_before"),
        "mean_after": _avg("mean_after"),
        "mean_gain_pct": _avg("mean_gain_pct"),
        "brisque_before": _avg("brisque_before"),
        "brisque_after": _avg("brisque_after"),
        "brisque_gain_pct": _avg("brisque_gain_pct"),
        "niqe_before": _avg("niqe_before"),
        "niqe_after": _avg("niqe_after"),
        "niqe_gain_pct": _avg("niqe_gain_pct"),
        "piqe_before": _avg("piqe_before"),
        "piqe_after": _avg("piqe_after"),
        "piqe_gain_pct": _avg("piqe_gain_pct"),
    }

    write_report(out_report, rows, avg)
    print(f"Готово. Видео: {out_video}")
    print(f"Отчет: {out_report}")


if __name__ == "__main__":
    main()

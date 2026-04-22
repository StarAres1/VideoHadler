import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
try:
    import torch_directml
except ImportError:
    torch_directml = None

try:
    from skimage.metrics import structural_similarity as ssim_skimage
except ImportError:
    ssim_skimage = None


# ---------------------------
# Пути и гиперпараметры
# ---------------------------
INPUT_DIR = Path(r"C:\Projects\QtTrial\dataset\half")
REFERENCE_DIR = Path(r"C:\Projects\QtTrial\dataset\half_ref")
OUTPUT_DIR = Path(r"C:\Projects\QtTrial\app\neural_network\img2img\image2image_runs")

TARGET_WIDTH = 700
TARGET_HEIGHT = 400
TARGET_SIZE_HW = (TARGET_HEIGHT, TARGET_WIDTH)

TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_SEED = 42

BATCH_SIZE = 6
NUM_EPOCHS = 40
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.9
NUM_WORKERS = 0
RESUME_TRAINING = True

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    # По запросу пользователя: приоритет DirectML, затем CPU.
    if torch_directml is not None:
        return torch_directml.device()
    return torch.device("cpu")


def list_image_files(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXT])


def to_landscape(img: Image.Image) -> Image.Image:
    if img.height > img.width:
        return img.transpose(Image.Transpose.ROTATE_90)
    return img


def preprocess_rgb(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = to_landscape(img)
        img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


class PairedImageDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        input_path, ref_path = self.pairs[idx]
        image_np = preprocess_rgb(input_path)
        ref_np = preprocess_rgb(ref_path)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
        ref_tensor = torch.from_numpy(ref_np).permute(2, 0, 1).contiguous()

        return image_tensor, ref_tensor, input_path.name


class ResUNetEnhancer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Готовый ResUNet: U-Net с энкодером ResNet18.
        # in_channels=3 (RGB вход), classes=3 (RGB выход).
        self.model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=3,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ограничиваем выход диапазоном [0, 1], т.к. референсы нормализованы в [0, 1].
        return torch.sigmoid(self.model(x))


def rms_contrast_rgb(image_chw: np.ndarray) -> float:
    # image_chw: [3, H, W] в диапазоне [0, 1]
    rgb = np.transpose(image_chw, (1, 2, 0))
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return float(np.sqrt(np.mean((gray - np.mean(gray)) ** 2)))


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1e-12:
        return 100.0
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def fallback_ssim_global(pred: np.ndarray, target: np.ndarray) -> float:
    pred_gray = 0.299 * pred[0] + 0.587 * pred[1] + 0.114 * pred[2]
    target_gray = 0.299 * target[0] + 0.587 * target[1] + 0.114 * target[2]

    mu_x = float(np.mean(pred_gray))
    mu_y = float(np.mean(target_gray))
    sigma_x = float(np.var(pred_gray))
    sigma_y = float(np.var(target_gray))
    sigma_xy = float(np.mean((pred_gray - mu_x) * (target_gray - mu_y)))

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return float(numerator / (denominator + 1e-12))


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    if ssim_skimage is None:
        return fallback_ssim_global(pred, target)
    pred_hwc = np.transpose(pred, (1, 2, 0))
    target_hwc = np.transpose(target, (1, 2, 0))
    return float(ssim_skimage(pred_hwc, target_hwc, data_range=1.0, channel_axis=2))


def build_pairs(input_dir: Path, reference_dir: Path) -> List[Tuple[Path, Path]]:
    input_files = list_image_files(input_dir)
    reference_by_stem = {p.stem: p for p in list_image_files(reference_dir)}
    pairs: List[Tuple[Path, Path]] = []

    for inp in input_files:
        # Правило соответствия:
        # референс: "<number>.<ext>", вход: "<same_number>_<anything>.<ext>"
        base_part = inp.stem.split("_", 1)[0]
        if base_part in reference_by_stem:
            pairs.append((inp, reference_by_stem[base_part]))

    return pairs


def evaluate_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, refs, _ in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            refs = refs.to(device)
            outputs = model(images)
            loss = criterion(outputs, refs)
            total_loss += loss.item() * images.size(0)
    return total_loss / max(1, len(loader.dataset))


def save_test_report(
    report_path: Path,
    rows: List[dict],
    avg_metrics: dict,
    test_loss: float,
) -> None:
    with report_path.open("w", encoding="utf-8") as f:
        f.write("ОТЧЕТ ПО ТЕСТОВОЙ ВЫБОРКЕ (ResNet18 image-to-image)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Размер изображения: {TARGET_WIDTH}x{TARGET_HEIGHT}, RGB\n")
        f.write("-" * 80 + "\n\n")

        f.write("По каждому изображению:\n")
        for row in rows:
            f.write(
                f"{row['name']}: "
                f"SSIM={row['ssim']:.6f}, "
                f"PSNR={row['psnr']:.4f}, "
                f"RMS(pred)={row['rms_pred']:.6f}, "
                f"RMS(ref)={row['rms_ref']:.6f}, "
                f"RMS(input)={row['rms_input']:.6f}, "
                f"RMS increase %={row['rms_increase_pct']:.2f}\n"
            )

        f.write("\n" + "-" * 80 + "\n")
        f.write("Средние метрики по тестовой выборке:\n")
        f.write(f"Test loss (MSE): {test_loss:.6f}\n")
        f.write(f"Mean SSIM: {avg_metrics['ssim']:.6f}\n")
        f.write(f"Mean PSNR: {avg_metrics['psnr']:.4f}\n")
        f.write(f"Mean RMS(pred): {avg_metrics['rms_pred']:.6f}\n")
        f.write(f"Mean RMS(ref): {avg_metrics['rms_ref']:.6f}\n")
        f.write(f"Mean RMS(input): {avg_metrics['rms_input']:.6f}\n")
        f.write(f"Mean RMS increase %: {avg_metrics['rms_increase_pct']:.2f}\n")


def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    history: List[dict],
    best_val_loss: float,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, checkpoint_path)


def save_epoch_report(report_path: Path, history: List[dict], best_val_loss: float) -> None:
    with report_path.open("w", encoding="utf-8") as f:
        f.write("ОТЧЕТ ПО ЭПОХАМ ОБУЧЕНИЯ\n")
        f.write("=" * 80 + "\n")
        f.write(f"Дата обновления: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Всего записей: {len(history)}\n")
        f.write(f"Текущий лучший val_loss: {best_val_loss:.6f}\n")
        f.write("-" * 80 + "\n\n")

        for row in history:
            f.write(
                f"epoch={row['epoch']}, "
                f"train_loss={row['train_loss']:.6f}, "
                f"val_loss={row['val_loss']:.6f}\n"
            )


def save_split(split_path: Path, train_pairs, val_pairs, test_pairs) -> None:
    payload = {
        "train": [{"input": str(inp), "reference": str(ref)} for inp, ref in train_pairs],
        "val": [{"input": str(inp), "reference": str(ref)} for inp, ref in val_pairs],
        "test": [{"input": str(inp), "reference": str(ref)} for inp, ref in test_pairs],
    }
    with split_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_split(split_path: Path):
    with split_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    def to_pairs(items):
        return [(Path(row["input"]), Path(row["reference"])) for row in items]

    return to_pairs(payload["train"]), to_pairs(payload["val"]), to_pairs(payload["test"])


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Папка с входными изображениями не найдена: {INPUT_DIR}")
    if not REFERENCE_DIR.exists():
        raise FileNotFoundError(f"Папка с референсами не найдена: {REFERENCE_DIR}")

    set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_models_dir = OUTPUT_DIR / "all_models"
    all_models_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(INPUT_DIR, REFERENCE_DIR)
    if len(pairs) < 10:
        raise RuntimeError(
            f"Слишком мало пар для обучения ({len(pairs)}). "
            f"Проверьте шаблон имен: input '<num>_*.ext' и reference '<num>.ext'."
        )

    split_path = OUTPUT_DIR / "split.json"
    if split_path.exists():
        train_pairs, val_pairs, test_pairs = load_split(split_path)
        print(f"Загружено разбиение из {split_path}")
    else:
        train_pairs, test_pairs = train_test_split(pairs, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        effective_val_size = VAL_SIZE / (1.0 - TEST_SIZE)
        train_pairs, val_pairs = train_test_split(
            train_pairs,
            test_size=effective_val_size,
            random_state=RANDOM_SEED,
        )
        save_split(split_path, train_pairs, val_pairs, test_pairs)
        print(f"Разбиение сохранено в {split_path}")

    train_ds = PairedImageDataset(train_pairs)
    val_ds = PairedImageDataset(val_pairs)
    test_ds = PairedImageDataset(test_pairs)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    device = get_device()
    print(f"Device: {device}")
    if torch_directml is None:
        print("torch_directml не установлен, используется CPU.")
    print(f"Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    model = ResUNetEnhancer().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    history = []
    best_val_loss = float("inf")
    best_model_path = OUTPUT_DIR / "best_model.pth"
    latest_checkpoint_path = OUTPUT_DIR / "latest_checkpoint.pth"
    epoch_report_path = OUTPUT_DIR / "epoch_report.txt"
    start_epoch = 1

    if RESUME_TRAINING and latest_checkpoint_path.exists():
        checkpoint = torch.load(latest_checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        history = checkpoint.get("history", [])
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"Возобновление обучения с эпохи {start_epoch}")
    else:
        print("Обучение с нуля")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for images, refs, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]", leave=False):
            images = images.to(device)
            refs = refs.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, refs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / max(1, len(train_loader.dataset))
        val_loss = evaluate_epoch(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  Новая лучшая модель сохранена: {best_model_path}")

        # Сохраняем модель каждой эпохи, чтобы иметь полный набор результатов обучения.
        epoch_model_path = all_models_dir / f"epoch_{epoch:03d}.pth"
        torch.save(model.state_dict(), epoch_model_path)

        save_epoch_report(epoch_report_path, history, best_val_loss)
        save_checkpoint(
            checkpoint_path=latest_checkpoint_path,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            history=history,
            best_val_loss=best_val_loss,
        )

    # Загрузка лучшей модели перед тестом
    best_state = torch.load(best_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_state)
    model.eval()

    rows = []
    test_loss_sum = 0.0
    with torch.no_grad():
        for images, refs, names in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            refs = refs.to(device)
            outputs = model(images)
            loss = criterion(outputs, refs)
            test_loss_sum += loss.item()

            pred_np = outputs[0].detach().cpu().numpy()
            ref_np = refs[0].detach().cpu().numpy()
            input_np = images[0].detach().cpu().numpy()

            rms_input = rms_contrast_rgb(input_np)
            rms_pred = rms_contrast_rgb(pred_np)
            rms_ref = rms_contrast_rgb(ref_np)
            rms_increase_pct = ((rms_pred - rms_input) / (rms_input + 1e-12)) * 100.0

            rows.append(
                {
                    "name": names[0],
                    "ssim": compute_ssim(pred_np, ref_np),
                    "psnr": psnr(pred_np, ref_np),
                    "rms_pred": rms_pred,
                    "rms_ref": rms_ref,
                    "rms_input": rms_input,
                    "rms_increase_pct": rms_increase_pct,
                }
            )

    test_loss = test_loss_sum / max(1, len(test_ds))
    avg_metrics = {
        "ssim": float(np.mean([r["ssim"] for r in rows])),
        "psnr": float(np.mean([r["psnr"] for r in rows])),
        "rms_pred": float(np.mean([r["rms_pred"] for r in rows])),
        "rms_ref": float(np.mean([r["rms_ref"] for r in rows])),
        "rms_input": float(np.mean([r["rms_input"] for r in rows])),
        "rms_increase_pct": float(np.mean([r["rms_increase_pct"] for r in rows])),
    }

    report_path = OUTPUT_DIR / "test_report.txt"
    save_test_report(report_path, rows, avg_metrics, test_loss)

    history_path = OUTPUT_DIR / "train_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"Готово. Отчет: {report_path}")
    print(f"История обучения: {history_path}")


if __name__ == "__main__":
    main()

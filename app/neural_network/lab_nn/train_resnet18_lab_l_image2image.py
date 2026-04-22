import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2
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

try:
    import pyiqa
except ImportError:
    pyiqa = None


# ---------------------------
# Пути и гиперпараметры
# ---------------------------
INPUT_DIR = Path(r"C:\Projects\QtTrial\dataset\half")
REFERENCE_DIR = Path(r"C:\Projects\QtTrial\dataset\half_ref")
OUTPUT_DIR = Path(r"C:\Projects\QtTrial\app\neural_network\lab_nn\lab_l_runs")

TARGET_WIDTH = 700
TARGET_HEIGHT = 400
TARGET_SIZE_HW = (TARGET_HEIGHT, TARGET_WIDTH)

TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_SEED = 42

BATCH_SIZE = 6
NUM_EPOCHS = 20
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
    if torch_directml is not None:
        return torch_directml.device()
    return torch.device("cpu")


def list_image_files(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXT])


def to_landscape(img: Image.Image) -> Image.Image:
    if img.height > img.width:
        return img.transpose(Image.Transpose.ROTATE_90)
    return img


def preprocess_l_channel(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = to_landscape(img)
        img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.BICUBIC)
        rgb = np.asarray(img, dtype=np.uint8)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel = lab[..., 0].astype(np.float32) / 255.0
    return l_channel


def _random_crop_same(img_a: np.ndarray, img_b: np.ndarray, min_scale: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img_a.shape
    scale = np.random.uniform(min_scale, 1.0)
    crop_w = max(32, int(w * scale))
    crop_h = max(32, int(h * scale))
    x0 = np.random.randint(0, max(1, w - crop_w + 1))
    y0 = np.random.randint(0, max(1, h - crop_h + 1))
    a = img_a[y0:y0 + crop_h, x0:x0 + crop_w]
    b = img_b[y0:y0 + crop_h, x0:x0 + crop_w]
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_CUBIC)
    b = cv2.resize(b, (w, h), interpolation=cv2.INTER_CUBIC)
    return a, b


def _random_rotate_same(img_a: np.ndarray, img_b: np.ndarray, max_deg: float = 7.0) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img_a.shape
    angle = float(np.random.uniform(-max_deg, max_deg))
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    a = cv2.warpAffine(img_a, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    b = cv2.warpAffine(img_b, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return a, b


def _augment_input_l(input_l: np.ndarray) -> np.ndarray:
    out = input_l.copy()

    if np.random.rand() < 0.30:
        alpha = float(np.random.uniform(0.92, 1.10))
        beta = float(np.random.uniform(-0.04, 0.04))
        out = np.clip(alpha * out + beta, 0.0, 1.0)

    return np.clip(out, 0.0, 1.0)


def augment_pair_train(input_l: np.ndarray, ref_l: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a, b = input_l, ref_l

    if np.random.rand() < 0.5:
        a = np.flip(a, axis=1).copy()
        b = np.flip(b, axis=1).copy()

    if np.random.rand() < 0.25:
        a = np.flip(a, axis=0).copy()
        b = np.flip(b, axis=0).copy()

    if np.random.rand() < 0.25:
        a, b = _random_crop_same(a, b, min_scale=0.9)

    if np.random.rand() < 0.20:
        a, b = _random_rotate_same(a, b, max_deg=4.0)

    # Only degrade input branch photometrically/noise-wise.
    a = _augment_input_l(a)
    return a, b


class PairedImageDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path]], is_train: bool = False) -> None:
        self.pairs = pairs
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        input_path, ref_path = self.pairs[idx]
        image_l = preprocess_l_channel(input_path)
        ref_l = preprocess_l_channel(ref_path)
        if self.is_train:
            image_l, ref_l = augment_pair_train(image_l, ref_l)

        image_tensor = torch.from_numpy(image_l).unsqueeze(0).contiguous()
        ref_tensor = torch.from_numpy(ref_l).unsqueeze(0).contiguous()
        return image_tensor, ref_tensor, input_path.name


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
    grad_sq = gx * gx + gy * gy
    return float(np.mean(grad_sq))


def image_entropy(image_hw: np.ndarray) -> float:
    image_u8 = np.clip(image_hw * 255.0, 0, 255).astype(np.uint8)
    hist = cv2.calcHist([image_u8], [0], None, [256], [0, 256]).ravel()
    p = hist / (np.sum(hist) + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


class NoReferenceIqaEvaluator:
    def __init__(self) -> None:
        self.available = False
        self.metric_brisque = None
        self.metric_niqe = None
        self.metric_piqe = None

        if pyiqa is None:
            return
        try:
            # CPU mode to keep compatibility with DirectML training flow.
            self.metric_brisque = pyiqa.create_metric("brisque", device="cpu")
            self.metric_niqe = pyiqa.create_metric("niqe", device="cpu")
            self.metric_piqe = pyiqa.create_metric("piqe", device="cpu")
            self.available = True
        except Exception:
            self.available = False

    def evaluate_l(self, image_hw: np.ndarray) -> Tuple[float, float, float]:
        if not self.available:
            nan = float("nan")
            return nan, nan, nan

        # pyiqa expects NCHW tensor in [0, 1]. For L-channel repeat to pseudo-RGB.
        tensor = torch.from_numpy(image_hw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        tensor_rgb = tensor.repeat(1, 3, 1, 1)
        with torch.no_grad():
            brisque = float(self.metric_brisque(tensor_rgb).item())
            niqe = float(self.metric_niqe(tensor_rgb).item())
            piqe = float(self.metric_piqe(tensor_rgb).item())
        return brisque, niqe, piqe


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1e-12:
        return 100.0
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def fallback_ssim_global(pred: np.ndarray, target: np.ndarray) -> float:
    mu_x = float(np.mean(pred))
    mu_y = float(np.mean(target))
    sigma_x = float(np.var(pred))
    sigma_y = float(np.var(target))
    sigma_xy = float(np.mean((pred - mu_x) * (target - mu_y)))
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return float(numerator / (denominator + 1e-12))


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    if ssim_skimage is None:
        return fallback_ssim_global(pred, target)
    return float(ssim_skimage(pred, target, data_range=1.0))


def gradient_map(t: torch.Tensor) -> torch.Tensor:
    # t: [B, 1, H, W]
    sobel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        dtype=t.dtype,
        device=t.device,
    ).unsqueeze(1)
    sobel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        dtype=t.dtype,
        device=t.device,
    ).unsqueeze(1)
    gx = torch.nn.functional.conv2d(t, sobel_x, padding=1)
    gy = torch.nn.functional.conv2d(t, sobel_y, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


def ssim_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Global SSIM approximation over full image for stability and simplicity.
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = pred.mean(dim=(2, 3), keepdim=True)
    mu_y = target.mean(dim=(2, 3), keepdim=True)
    sigma_x = ((pred - mu_x) ** 2).mean(dim=(2, 3), keepdim=True)
    sigma_y = ((target - mu_y) ** 2).mean(dim=(2, 3), keepdim=True)
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean(dim=(2, 3), keepdim=True)
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / (denominator + 1e-8)
    return ssim_map.mean()


class CompositeEnhancementLoss(nn.Module):
    def __init__(
        self,
        w_l1: float = 0.45,
        w_ssim: float = 0.25,
        w_grad: float = 0.25,
        w_mean: float = 0.05,
    ) -> None:
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_grad = w_grad
        self.w_mean = w_mean
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1_loss = self.l1(pred, target)
        ssim_loss = 1.0 - ssim_torch(pred, target)
        grad_loss = torch.mean(torch.abs(gradient_map(pred) - gradient_map(target)))
        mean_loss = torch.mean(torch.abs(pred.mean(dim=(2, 3)) - target.mean(dim=(2, 3))))
        return (
            self.w_l1 * l1_loss
            + self.w_ssim * ssim_loss
            + self.w_grad * grad_loss
            + self.w_mean * mean_loss
        )


def build_pairs(input_dir: Path, reference_dir: Path) -> List[Tuple[Path, Path]]:
    input_files = list_image_files(input_dir)
    reference_by_stem = {p.stem: p for p in list_image_files(reference_dir)}
    pairs: List[Tuple[Path, Path]] = []
    for inp in input_files:
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


def save_test_report(report_path: Path, rows: List[dict], avg_metrics: dict, test_loss: float) -> None:
    with report_path.open("w", encoding="utf-8") as f:
        f.write("ОТЧЕТ ПО ТЕСТОВОЙ ВЫБОРКЕ (ResUNet LAB-L image-to-image)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Размер изображения: {TARGET_WIDTH}x{TARGET_HEIGHT}, L-channel\n")
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
                f"RMS increase %={row['rms_increase_pct']:.2f}, "
                f"LapVar(pred)={row['lap_var_pred']:.4f}, "
                f"Tenengrad(pred)={row['tenengrad_pred']:.4f}, "
                f"Entropy(pred)={row['entropy_pred']:.4f}, "
                f"MeanL(pred)={row['mean_l_pred']:.4f}, "
                f"BRISQUE(pred)={row['brisque_pred']:.4f}, "
                f"NIQE(pred)={row['niqe_pred']:.4f}, "
                f"PIQE(pred)={row['piqe_pred']:.4f}\n"
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
        f.write(f"Mean LapVar(pred): {avg_metrics['lap_var_pred']:.4f}\n")
        f.write(f"Mean Tenengrad(pred): {avg_metrics['tenengrad_pred']:.4f}\n")
        f.write(f"Mean Entropy(pred): {avg_metrics['entropy_pred']:.4f}\n")
        f.write(f"Mean MeanL(pred): {avg_metrics['mean_l_pred']:.4f}\n")
        f.write(f"Mean BRISQUE(pred): {avg_metrics['brisque_pred']:.4f}\n")
        f.write(f"Mean NIQE(pred): {avg_metrics['niqe_pred']:.4f}\n")
        f.write(f"Mean PIQE(pred): {avg_metrics['piqe_pred']:.4f}\n")


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

    train_ds = PairedImageDataset(train_pairs, is_train=True)
    val_ds = PairedImageDataset(val_pairs, is_train=False)
    test_ds = PairedImageDataset(test_pairs, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    device = get_device()
    print(f"Device: {device}")
    if torch_directml is None:
        print("torch_directml не установлен, используется CPU.")
    print(f"Train/Val/Test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    model = ResUNetEnhancerL().to(device)
    iqa_evaluator = NoReferenceIqaEvaluator()
    if not iqa_evaluator.available:
        print("pyiqa не найден/не инициализирован: BRISQUE/NIQE/PIQE будут записаны как nan.")

    criterion = CompositeEnhancementLoss()
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

            pred_l = outputs[0, 0].detach().cpu().numpy()
            ref_l = refs[0, 0].detach().cpu().numpy()
            input_l = images[0, 0].detach().cpu().numpy()

            rms_input = rms_contrast_l(input_l)
            rms_pred = rms_contrast_l(pred_l)
            rms_ref = rms_contrast_l(ref_l)
            rms_increase_pct = ((rms_pred - rms_input) / (rms_input + 1e-12)) * 100.0
            lap_var_pred = laplacian_variance(pred_l)
            tenengrad_pred = tenengrad_sharpness(pred_l)
            entropy_pred = image_entropy(pred_l)
            mean_l_pred = float(np.mean(pred_l))
            brisque_pred, niqe_pred, piqe_pred = iqa_evaluator.evaluate_l(pred_l)

            rows.append(
                {
                    "name": names[0],
                    "ssim": compute_ssim(pred_l, ref_l),
                    "psnr": psnr(pred_l, ref_l),
                    "rms_pred": rms_pred,
                    "rms_ref": rms_ref,
                    "rms_input": rms_input,
                    "rms_increase_pct": rms_increase_pct,
                    "lap_var_pred": lap_var_pred,
                    "tenengrad_pred": tenengrad_pred,
                    "entropy_pred": entropy_pred,
                    "mean_l_pred": mean_l_pred,
                    "brisque_pred": brisque_pred,
                    "niqe_pred": niqe_pred,
                    "piqe_pred": piqe_pred,
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
        "lap_var_pred": float(np.mean([r["lap_var_pred"] for r in rows])),
        "tenengrad_pred": float(np.mean([r["tenengrad_pred"] for r in rows])),
        "entropy_pred": float(np.mean([r["entropy_pred"] for r in rows])),
        "mean_l_pred": float(np.mean([r["mean_l_pred"] for r in rows])),
        "brisque_pred": float(np.nanmean([r["brisque_pred"] for r in rows])),
        "niqe_pred": float(np.nanmean([r["niqe_pred"] for r in rows])),
        "piqe_pred": float(np.nanmean([r["piqe_pred"] for r in rows])),
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

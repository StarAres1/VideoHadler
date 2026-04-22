import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


def read_history(history_path: Path):
    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)
    if not isinstance(history, list) or len(history) == 0:
        raise ValueError(f"Некорректный train_history.json: {history_path}")
    return history


def parse_test_report(report_path: Path):
    if not report_path.exists():
        return {}
    text = report_path.read_text(encoding="utf-8")
    patterns = {
        "test_loss": r"Test loss \(MSE\):\s*([0-9]*\.?[0-9]+)",
        "mean_ssim": r"Mean SSIM:\s*([0-9]*\.?[0-9]+)",
        "mean_psnr": r"Mean PSNR:\s*([0-9]*\.?[0-9]+)",
        "mean_rms_increase_pct": r"Mean RMS increase %:\s*([-+]?[0-9]*\.?[0-9]+)",
    }
    out = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            out[key] = float(match.group(1))
    return out


def plot_losses(history, output_dir: Path):
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, marker="o", label="Train Loss")
    plt.plot(epochs, val_loss, marker="o", label="Val Loss")
    plt.title("Train / Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_path = output_dir / "loss_curves.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_general_metrics(test_metrics: dict, output_dir: Path):
    if not test_metrics:
        return None

    labels = []
    values = []
    for key in ("mean_ssim", "mean_psnr", "mean_rms_increase_pct", "test_loss"):
        if key in test_metrics:
            labels.append(key)
            values.append(test_metrics[key])

    if not labels:
        return None

    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, values)
    plt.title("Test Summary Metrics")
    plt.ylabel("Value")
    plt.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    save_path = output_dir / "test_summary_metrics.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def main():
    parser = argparse.ArgumentParser(
        description="Построение графиков метрик обучения image-to-image модели."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(r"C:\Projects\QtTrial\app\neural_network\models\change_cool\image2image_runs"),
        help="Папка запуска, где лежат train_history.json и test_report.txt",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    history_path = run_dir / "train_history.json"
    report_path = run_dir / "test_report.txt"
    output_dir = run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    history = read_history(history_path)
    test_metrics = parse_test_report(report_path)

    loss_plot = plot_losses(history, output_dir)
    test_plot = plot_general_metrics(test_metrics, output_dir)

    print(f"График train/val loss сохранен: {loss_plot}")
    if test_plot is not None:
        print(f"График сводных тестовых метрик сохранен: {test_plot}")
    else:
        print("Сводные тестовые метрики не построены (не найден test_report.txt или метрики в нем).")


if __name__ == "__main__":
    main()

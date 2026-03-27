import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def compute_class_accuracy(confusion_matrix):
    """
    Вычисляет accuracy для каждого класса на основе матрицы ошибок.
    Для класса i: cm[i][i] / sum(cm[i][:])
    Возвращает список из 6 значений (для классов 0..5).
    """
    cm = np.array(confusion_matrix)
    # Сумма по строкам (все реальные образцы класса)
    row_sums = cm.sum(axis=1)
    # Диагональ (правильные предсказания)
    diag = np.diag(cm)
    # Избегаем деления на ноль (если класс отсутствует)
    acc = np.divide(diag, row_sums, out=np.zeros_like(diag, dtype=float), where=row_sums != 0)
    return acc.tolist()


def plot_balanced_accuracy(epochs, values, ax):
    ax.plot(epochs, values, marker='o', linestyle='-', color='b')
    ax.set_title('Balanced Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Balanced Accuracy')
    ax.grid(True)


def plot_per_class_accuracy(epochs, class_accuracies, ax):
    # class_accuracies: список списков, каждый внутренний список содержит accuracy для всех классов на одной эпохе
    # Преобразуем для удобства: для каждого класса собираем значения по эпохам
    class_acc_by_epoch = np.array(class_accuracies).T  # теперь строки = классы, столбцы = эпохи
    for cls_idx, acc_vals in enumerate(class_acc_by_epoch):
        ax.plot(epochs, acc_vals, marker='.', linestyle='-', label=f'Class {cls_idx}')
    ax.set_title('Per‑Class Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='best')
    ax.grid(True)


def plot_losses(epochs, train_losses, val_losses, ax):
    ax.plot(epochs, train_losses, marker='o', linestyle='-', color='r', label='Train Loss')
    ax.plot(epochs, val_losses, marker='o', linestyle='-', color='g', label='Val Loss')
    ax.set_title('Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(loc='best')
    ax.grid(True)


def plot_macro_f1(epochs, macro_f1, ax):
    ax.plot(epochs, macro_f1, marker='o', linestyle='-', color='m')
    ax.set_title('Macro F1 Score')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.grid(True)


def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from JSON log file.')
    parser.add_argument('--file', type=str, default='logs/training_log.json',
                        help='Path to JSON log file (default: logs.json)')
    args = parser.parse_args()

    # Загрузка данных
    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    except Exception as e:
        print(f'Error loading file: {e}')
        return

    # Извлечение данных по эпохам
    epochs = [entry['epoch'] for entry in logs]
    train_loss = [entry['train_loss'] for entry in logs]
    val_loss = [entry['val_loss'] for entry in logs]
    balanced_acc = [entry['val_balanced_accuracy'] for entry in logs]
    macro_f1 = [entry['val_macro_f1'] for entry in logs]

    # Вычисление точности для каждого класса
    class_accuracies = []
    for entry in logs:
        cm = entry['val_confusion_matrix']
        class_acc = compute_class_accuracy(cm)
        class_accuracies.append(class_acc)

    # Создание подграфиков (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training Metrics', fontsize=16)

    plot_balanced_accuracy(epochs, balanced_acc, axes[0, 0])
    plot_per_class_accuracy(epochs, class_accuracies, axes[0, 1])
    plot_losses(epochs, train_loss, val_loss, axes[1, 0])
    plot_macro_f1(epochs, macro_f1, axes[1, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # учёт заголовка
    plt.show()


if __name__ == '__main__':
    main()
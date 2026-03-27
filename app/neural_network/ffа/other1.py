import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch_directml

from app.neural_network.ffа.Dataset1 import ImageDataset


# ---------------------------
# Класс EarlyStopping
# ---------------------------
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = None
        self.best_weights = None
        self.stop = False

    def __call__(self, model, val_loss, epoch):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict()
            print(f"  ✅ Улучшение: val_loss = {val_loss:.4f}")
        else:
            self.counter += 1
            print(f"  ⏳ Нет улучшений: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                    print("  🔄 Восстановлены веса лучшей эпохи")
                print(f"  EarlyStopping: остановка на эпохе {epoch}")


# ---------------------------
# Гиперпараметры
# ---------------------------
data_root = "../../../other/scripts/organized_images"
batch_size = 64
num_epochs = 100
learning_rate = 0.0003
random_seed = 10
model_save_dir = "./saved_models"
logs_dir = "./logs"

os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# ---------------------------
# Трансформации
# ---------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

transform_val_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---------------------------
# Загрузка датасета
# ---------------------------
print(f"Текущая директория: {os.getcwd()}")
print(f"Путь к данным: {os.path.abspath(data_root)}")

try:
    full_dataset = ImageDataset(root=data_root, transform=transform_train, verbose=True)
except (FileNotFoundError, RuntimeError) as e:
    print(f"\n❌ Ошибка: {e}")
    exit(1)

num_classes = len(full_dataset.classes)
labels = full_dataset.labels

# ---------------------------
# 📝 СОХРАНЕНИЕ МАППИНГА КЛАССОВ (НОВОЕ!)
# ---------------------------
mapping_file = os.path.join(logs_dir, "class_mapping.txt")
with open(mapping_file, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("СООТВЕТСТВИЕ ВЫХОДОВ НЕЙРОННОЙ СЕТИ И КЛАССОВ\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Всего классов: {num_classes}\n")
    f.write(f"Дата создания: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("-" * 60 + "\n")
    f.write(f"{'Выход сети':<15} {'Индекс':<10} {'Класс (метод)'}\n")
    f.write("-" * 60 + "\n")

    # Сортируем по индексу для удобства
    sorted_classes = sorted(full_dataset.class_to_idx.items(), key=lambda x: x[1])
    for class_name, idx in sorted_classes:
        f.write(f"{idx:<15} {idx:<10} {class_name}\n")

    f.write("-" * 60 + "\n")
    f.write("\nПример использования:\n")
    f.write("  Если модель вернула предсказание = 5, то это класс: " +
            (sorted_classes[5][0] if len(sorted_classes) > 5 else "N/A") + "\n")
    f.write("=" * 60 + "\n")

print(f"✅ Маппинг классов сохранен: {mapping_file}")

# ---------------------------
# Разбиение (стратифицированное)
# ---------------------------
train_idx, temp_idx = train_test_split(
    np.arange(len(full_dataset)),
    test_size=0.3,
    stratify=labels,
    random_state=random_seed
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    stratify=[labels[i] for i in temp_idx],
    random_state=random_seed
)


def make_subset(indices, transform):
    ds = ImageDataset(root=data_root, transform=transform, verbose=False)
    return Subset(ds, indices)


train_dataset = make_subset(train_idx, transform_train)
val_dataset = make_subset(val_idx, transform_val_test)
test_dataset = make_subset(test_idx, transform_val_test)

print(f"\nРазмеры выборок:")
print(f"  Train: {len(train_dataset)}")
print(f"  Val:   {len(val_dataset)}")
print(f"  Test:  {len(test_dataset)}")

# ---------------------------
# DataLoaders
# ---------------------------
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ---------------------------
# Модель и устройство
# ---------------------------
device = torch_directml.device()
print(f"\nИспользуется устройство: {device}")

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.001)

# Планировщик скорости обучения
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Ранняя остановка
early_stopping = EarlyStopping(patience=15, restore_best_weights=True)

# ---------------------------
# Логирование метрик
# ---------------------------
metrics_log = []

# ---------------------------
# Цикл обучения
# ---------------------------
print("\n" + "=" * 50)
print("НАЧАЛО ОБУЧЕНИЯ")
print("=" * 50 + "\n")

for epoch in range(num_epochs):
    # --- TRAIN ---
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        progress_bar.set_postfix(loss=loss.item())

    train_loss = running_loss / len(train_dataset)

    # --- VALIDATION ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    val_progress = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]  ')
    with torch.no_grad():
        for inputs, labels in val_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss / len(val_dataset)
    val_acc = val_correct / val_total

    print(f'\nEpoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}')

    # Scheduler
    scheduler.step(val_loss)

    # EarlyStopping
    early_stopping(model, val_loss, epoch + 1)

    # Сохранение метрик
    epoch_metrics = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    }
    metrics_log.append(epoch_metrics)
    with open(os.path.join(logs_dir, "training_log1.json"), "w") as f:
        json.dump(metrics_log, f, indent=4)

    # Сохранение модели
    model_path = os.path.join(model_save_dir, f"model_epoch_{epoch + 1}.pth")
    torch.save(model.state_dict(), model_path)

    val_progress.set_postfix(loss=val_loss, acc=val_acc)

    if early_stopping.stop:
        print("\n" + "=" * 50)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО (EarlyStopping)")
        print("=" * 50)
        break

print("\nОбучение завершено")

# ---------------------------
# Тестирование
# ---------------------------
print("\nЗапуск тестирования...")
model.eval()
test_correct = 0
test_total = 0

test_progress = tqdm(test_loader, desc='Testing')
with torch.no_grad():
    for inputs, labels in test_progress:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

test_acc = test_correct / test_total
print(f'\nTest Accuracy: {test_acc:.4f}')

final_log = {
    "test_accuracy": test_acc,
    "training_history": metrics_log
}
with open(os.path.join(logs_dir, "final_log1.json"), "w") as f:
    json.dump(final_log, f, indent=4)

print(f"Финальный лог сохранен в {logs_dir}/final_log1.json")
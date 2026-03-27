import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch_directml
from app.neural_network.Dataset import ImageDataset
# ---------------------------
# Гиперпараметры
# ---------------------------
# УКАЖИТЕ ПРАВИЛЬНЫЙ ПУТЬ К ВАШИМ ДАННЫМ (абсолютный или корректный относительный)
data_root = "../../other/scripts/organized_images"  # ← ИЗМЕНИТЕ ПРИ НЕОБХОДИМОСТИ
batch_size = 32
num_epochs = 100
learning_rate = 0.001
num_classes = 20  # Количество выходных нейронов (должно совпадать с количеством классов)
random_seed = 42
model_save_dir = "./saved_models"
logs_dir = "./logs"
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

# ---------------------------
# Трансформации
# ---------------------------
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

transform_val_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------------------
# Вывод текущей директории для отладки
# ---------------------------
print(f"Текущая рабочая директория: {os.getcwd()}")
print(f"Ожидаемый путь к данным: {os.path.abspath(data_root)}")

# ---------------------------
# Загрузка датасета
# ---------------------------
try:
    full_dataset = ImageDataset(root=data_root, transform=transform_train, verbose=True)
except (FileNotFoundError, RuntimeError) as e:
    print(f"\n❌ Ошибка: {e}")
    print("\nПожалуйста, проверьте:")
    print(f"1. Путь к данным: {data_root}")
    print("2. Структура папок должна быть:")
    print(f"   {data_root}/")
    print("      ├── CLAHE_2.0_4_4/")
    print("      │     ├── image1.jpg")
    print("      │     └── ...")
    print("      ├── gamma_0.86/")
    print("      └── ... (всего 20 папок)")
    print("3. Если путь относительный, убедитесь, что скрипт запускается из нужной директории.")
    print("   Текущая директория:", os.getcwd())
    exit(1)

# Проверяем, что количество классов соответствует ожидаемому
if len(full_dataset.classes) != num_classes:
    print(f"\n⚠️ Внимание: обнаружено {len(full_dataset.classes)} классов, а num_classes установлено в {num_classes}")
    print("Будет использовано фактическое количество классов.")
    num_classes = len(full_dataset.classes)
    # Обновляем выходной слой позже

labels = full_dataset.labels

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
val_dataset   = make_subset(val_idx, transform_val_test)
test_dataset  = make_subset(test_idx, transform_val_test)

print(f"\nРазмеры выборок:")
print(f"  Train: {len(train_dataset)}")
print(f"  Val:   {len(val_dataset)}")
print(f"  Test:  {len(test_dataset)}")

# ---------------------------
# DataLoaders
# ---------------------------
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ---------------------------
# Модель и устройство
# ---------------------------
device = torch_directml.device()
print(f"\nИспользуется устройство: {device}")

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # теперь num_classes корректен
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

# ---------------------------
# Логирование метрик
# ---------------------------
metrics_log = []

# ---------------------------
# Цикл обучения
# ---------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
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

    # Валидация
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    val_loss = val_loss / len(val_dataset)
    val_acc = val_correct / val_total

    print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}')

    # Сохранение метрик
    epoch_metrics = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    }
    metrics_log.append(epoch_metrics)
    with open(os.path.join(logs_dir, "training_log.json"), "w") as f:
        json.dump(metrics_log, f, indent=4)

    # Сохранение модели
    model_path = os.path.join(model_save_dir, f"model_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Модель сохранена: {model_path}")

print("Обучение завершено")

# ---------------------------
# Тестирование
# ---------------------------
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)
test_acc = test_correct / test_total
print(f'Test Accuracy: {test_acc:.4f}')

final_log = {
    "test_accuracy": test_acc,
    "training_history": metrics_log
}
with open(os.path.join(logs_dir, "final_log.json"), "w") as f:
    json.dump(final_log, f, indent=4)
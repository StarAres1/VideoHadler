import os
import random
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import torch_directml

# ---------------------------
# Класс ImageDataset (только grayscale, без конвертации в RGB)
# ---------------------------
class ImageDataset(data.Dataset):
    def __init__(self, root, transform=None, verbose=True):
        self.root = root
        self.transform = transform
        self.classes = []
        self.class_to_idx = {}
        self.files = []

        if not os.path.exists(root):
            raise FileNotFoundError(f"Папка не найдена: {root}")

        if verbose:
            print(f"Сканируем папку: {root}")

        for dir_name in sorted(os.listdir(root)):
            dir_path = os.path.join(root, dir_name)
            if not os.path.isdir(dir_path):
                continue

            idx = len(self.classes)
            self.class_to_idx[dir_name] = idx
            self.classes.append(dir_name)

            count = 0
            for file_name in sorted(os.listdir(dir_path)):
                file_path = os.path.join(dir_path, file_name)
                if os.path.isfile(file_path):
                    self.files.append((file_path, idx))
                    count += 1

            if verbose:
                print(f"  Найден класс: {dir_name} ({count} файлов)")

        self.labels = [label for _, label in self.files]

        if len(self.files) == 0:
            raise RuntimeError(f"В папке {root} не найдено изображений.")

        if verbose:
            print(f"Всего классов: {len(self.classes)}")
            print(f"Всего изображений: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        img = Image.open(path)
        img = img.convert('L')   # grayscale, один канал
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------
# Класс EarlyStopping (без изменений)
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
                self.best_weights = model.state_dict().copy()
            print(f"  ✅ Улучшение: val_loss = {val_loss:.4f}")
        else:
            self.counter += 1
            print(f"  ⏳ Нет улучшений: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    print("  🔄 Восстановлены веса лучшей эпохи")
                print(f"  🛑 EarlyStopping: остановка на эпохе {epoch}")


# ---------------------------
# Гиперпараметры и пути (без изменений)
# ---------------------------
data_root = "../../../other/scripts/organized_images"
batch_size = 64
num_epochs = 100
learning_rate = 0.001
momentum = 0.9
weight_decay = 0.0001
random_seed = 10
model_save_dir = "./saved_models"
logs_dir = "./logs"
checkpoint_dir = "./checkpoints"

os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

resume = True
checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
best_model_path = os.path.join(model_save_dir, "best_model.pth")
split_file = os.path.join(logs_dir, "split_indices.npz")

# ---------------------------
# Трансформации: добавляем аугментации, не влияющие на яркость/контраст
# ---------------------------
# Нормализация для одного канала (mean=0.5, std=0.5)
mean = [0.5]
std = [0.5]

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),               # сначала ресайз до базового размера
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),        # вертикальное отражение
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

transform_val_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# ---------------------------
# Загрузка полного датасета
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
# Сохранение маппинга классов (без изменений)
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
# Разбиение на train/val/test (без изменений)
# ---------------------------
if os.path.exists(split_file):
    print("Загрузка сохранённого разбиения...")
    split_data = np.load(split_file)
    train_idx = split_data['train_idx']
    val_idx = split_data['val_idx']
    test_idx = split_data['test_idx']
else:
    print("Выполняем стратифицированное разбиение...")
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
    np.savez(split_file, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    print("Разбиение сохранено.")

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
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# ---------------------------
# Устройство
# ---------------------------
device = torch_directml.device()
print(f"\nИспользуется устройство: {device}")

# ---------------------------
# Веса классов для функции потерь
# ---------------------------
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"\nВеса классов (тензор): {class_weights}")

# Выводим веса с привязкой к классам
print("\nСоответствие весов классов:")
sorted_classes = sorted(full_dataset.class_to_idx.items(), key=lambda x: x[1])
for idx, (class_name, _) in enumerate(sorted_classes):
    weight = class_weights[idx].item()
    print(f"  Класс {idx}: '{class_name}' -> вес = {weight:.4f}")

criterion = nn.CrossEntropyLoss(weight=class_weights)

# ---------------------------
# Модель ResNet18, адаптированная под 1 канал
# ---------------------------
def create_resnet18_grayscale(num_classes):
    # Загружаем предобученную модель (3 канала)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Заменяем входной слой на слой с 1 каналом
    original_conv1 = model.conv1
    new_conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=original_conv1.bias is not None
    )
    # Инициализируем веса как среднее по трём каналам
    with torch.no_grad():
        new_conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
        if original_conv1.bias is not None:
            new_conv1.bias.data = original_conv1.bias.data
    model.conv1 = new_conv1
    # Заменяем последний полносвязный слой
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

model = create_resnet18_grayscale(num_classes)
model = model.to(device)

# ---------------------------
# Оптимизатор и планировщик
# ---------------------------
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
early_stopping = EarlyStopping(patience=15, restore_best_weights=True)

# ---------------------------
# Загрузка чекпоинта (если нужно)
# ---------------------------
start_epoch = 0
metrics_log = []

if resume and os.path.exists(checkpoint_path):
    print(f"Загрузка чекпоинта из {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    early_stopping.best_loss = checkpoint['early_stopping_state']['best_loss']
    early_stopping.counter = checkpoint['early_stopping_state']['counter']
    early_stopping.best_weights = checkpoint['early_stopping_state']['best_weights']
    metrics_log = checkpoint['metrics_log']
    start_epoch = checkpoint['epoch']

    torch.set_rng_state(checkpoint['rng_state']['torch'])
    if torch.cuda.is_available() and checkpoint['rng_state']['cuda'] is not None:
        torch.cuda.set_rng_state_all(checkpoint['rng_state']['cuda'])
    np.random.set_state(checkpoint['rng_state']['numpy'])
    random.setstate(checkpoint['rng_state']['python'])

    print(f"✅ Обучение возобновляется с эпохи {start_epoch}")
else:
    print("🚀 Обучение начинается с нуля")
    metrics_log = []

# ---------------------------
# Функция сохранения чекпоинта (без изменений)
# ---------------------------
def save_checkpoint(epoch, model, optimizer, scheduler, early_stopping, metrics_log, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'early_stopping_state': {
            'best_loss': early_stopping.best_loss,
            'counter': early_stopping.counter,
            'best_weights': early_stopping.best_weights
        },
        'metrics_log': metrics_log,
        'random_seed': random_seed,
        'rng_state': {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy': np.random.get_state(),
            'python': random.getstate()
        }
    }
    torch.save(checkpoint, checkpoint_path)
    if is_best:
        torch.save(checkpoint, best_model_path)
    print(f"💾 Чекпоинт сохранён (эпоха {epoch})")

# ---------------------------
# Цикл обучения (без изменений)
# ---------------------------
print("\n" + "=" * 50)
print("НАЧАЛО ОБУЧЕНИЯ")
print("=" * 50 + "\n")

for epoch in range(start_epoch, num_epochs):
    # --- TRAIN ---
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

    # --- VALIDATION ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    val_progress = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]  ')
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

    current_lr = optimizer.param_groups[0]['lr']
    print(f'\nEpoch {epoch+1}: LR={current_lr:.6f}, Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}')

    scheduler.step(val_loss)
    early_stopping(model, val_loss, epoch+1)

    epoch_metrics = {
        "epoch": epoch+1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "learning_rate": current_lr
    }
    metrics_log.append(epoch_metrics)
    with open(os.path.join(logs_dir, "training_log.json"), "w") as f:
        json.dump(metrics_log, f, indent=4)

    save_checkpoint(epoch+1, model, optimizer, scheduler, early_stopping, metrics_log, is_best=False)
    val_progress.set_postfix(loss=val_loss, acc=val_acc)

    if early_stopping.stop:
        print("\n" + "=" * 50)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО (EarlyStopping)")
        print("=" * 50)
        break

print("\nОбучение завершено")

# ---------------------------
# Тестирование (без изменений)
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
with open(os.path.join(logs_dir, "final_log.json"), "w") as f:
    json.dump(final_log, f, indent=4)

print(f"Финальный лог сохранен в {logs_dir}/final_log.json")
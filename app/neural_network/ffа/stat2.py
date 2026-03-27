import json
import matplotlib.pyplot as plt

# 1. Загрузка данных из JSON файла
# Замените 'metrics.json' на имя вашего файла
with open('logs/training_log1.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. Извлечение данных для графиков
epochs = [item['epoch'] for item in data]
train_loss = [item['train_loss'] for item in data]
val_loss = [item['val_loss'] for item in data]
val_accuracy = [item['val_accuracy'] for item in data]

# 3. Создание фигуры с двумя подграфиками
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Первый график: Loss (train и val) ---
ax1.plot(epochs, train_loss, label='Train Loss', marker='o', color='blue')
ax1.plot(epochs, val_loss, label='Val Loss', marker='s', color='red')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Train Loss vs Val Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Второй график: Accuracy (val) ---
ax2.plot(epochs, val_accuracy, label='Val Accuracy', marker='o', color='green')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Validation Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)
# Опционально: форматирование оси Y для процентов (0.0 - 1.0)
ax2.set_ylim(0, 1)

# 4. Настройка компоновки и сохранение
plt.tight_layout()
plt.savefig('training_metrics1.png', dpi=300)
plt.show()

print("Графики успешно построены и сохранены в 'training_metrics1.png'")
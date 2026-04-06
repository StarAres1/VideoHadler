import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from app.core.ContrastImprover import ContrastImprover
import torch_directml


def create_resnet18_grayscale(num_classes):
    model = models.resnet18(weights=None)
    original_conv1 = model.conv1
    new_conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=original_conv1.bias is not None
    )
    with torch.no_grad():
        if original_conv1.weight is not None:
            new_conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
        if original_conv1.bias is not None:
            new_conv1.bias.data = original_conv1.bias.data
    model.conv1 = new_conv1
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------
# Параметры
# ---------------------------
input_folder = 'test_image'
output_folder = 'output'
model_path = 'last_balance_nn/saved_models/best_model.pth'
num_classes = 6
results_file = os.path.join(output_folder, 'results.txt')  # файл для сохранения результатов

class_mapping = {
    0: 'CLAHE_3.0_4_4',
    1: 'adjust_contrast_2.5_10',
    2: 'gamma_1.9',
    3: 'sigmoid+HE_0.3_12',
    4: 'sigmoid_0.3_12',
    5: 'sigmoid_0.4_12'
}

# ---------------------------
# Создание папок
# ---------------------------
os.makedirs(output_folder, exist_ok=True)

# ---------------------------
# Загрузка модели
# ---------------------------
print("Загрузка модели...")
device = torch_directml.device()
print(f"Используется устройство: {device}")

checkpoint = torch.load(model_path, map_location='cpu')
model = create_resnet18_grayscale(num_classes)

if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()


# ---------------------------
# Функция предобработки
# ---------------------------
def preprocess_image(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    pil_img = Image.fromarray(gray)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    tensor = transform(pil_img)
    return tensor.unsqueeze(0).to(device)


# ---------------------------
# Функция применения улучшений контраста и сохранения
# ---------------------------
def apply_and_save_contrast(original_rgb, original_filename):
    name_no_ext = os.path.splitext(original_filename)[0]

    img_adjust = ContrastImprover.adjust_contrast(original_rgb, alpha=2.5, beta=10)
    img_clahe = ContrastImprover.CLAHE(original_rgb, clipLimit=3.0, titleGridSizeX=4, titleGridSizeY=4)
    img_gamma = ContrastImprover.gamma_correction(original_rgb, gamma=1.9)
    img_sigmoid_03 = ContrastImprover.sigmoid_correction(original_rgb, cutoff=0.3, gain=12)
    img_sigmoid_04 = ContrastImprover.sigmoid_correction(original_rgb, cutoff=0.4, gain=12)
    img_sigmoid_tmp = ContrastImprover.sigmoid_correction(original_rgb, cutoff=0.3, gain=12)
    img_sigmoid_he = ContrastImprover.HE(img_sigmoid_tmp)

    cv2.imwrite(os.path.join(output_folder, f"{name_no_ext}_adjust_contrast_2.5_10.jpg"),
                cv2.cvtColor(img_adjust, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_folder, f"{name_no_ext}_CLAHE_3.0_4_4.jpg"),
                cv2.cvtColor(img_clahe, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_folder, f"{name_no_ext}_gamma_1.9.jpg"),
                cv2.cvtColor(img_gamma, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_folder, f"{name_no_ext}_sigmoid_0.3_12.jpg"),
                cv2.cvtColor(img_sigmoid_03, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_folder, f"{name_no_ext}_sigmoid_0.4_12.jpg"),
                cv2.cvtColor(img_sigmoid_04, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_folder, f"{name_no_ext}_sigmoid+HE_0.3_12.jpg"),
                cv2.cvtColor(img_sigmoid_he, cv2.COLOR_RGB2BGR))


# ---------------------------
# Основной цикл
# ---------------------------
supported_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_ext)]

if not files:
    print(f"В папке {input_folder} не найдено изображений.")
    exit(1)

print(f"\nНайдено файлов: {len(files)}")
print("Обработка...\n")

# Открываем файл для записи результатов
with open(results_file, 'w', encoding='utf-8') as res_file:
    res_file.write("Файл -> Индекс класса -> Название класса\n")
    res_file.write("-" * 60 + "\n")

    for filename in files:
        file_path = os.path.join(input_folder, filename)
        print(f"Обработка: {filename}")

        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            print(f"  Ошибка загрузки, пропускаем.")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 1. Применяем улучшения контраста и сохраняем
        apply_and_save_contrast(img_rgb, filename)

        # 2. Предсказание модели для исходного изображения
        input_tensor = preprocess_image(img_rgb)
        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item()

        class_name = class_mapping.get(predicted_class, "Неизвестный")

        # 3. Вывод в консоль и в файл в формате: файл -> номер класса -> название
        result_line = f"{filename} -> {predicted_class} -> {class_name}"
        print(f"  {result_line}")
        res_file.write(result_line + "\n")

print(f"\nГотово. Результаты сохранены в {results_file}")
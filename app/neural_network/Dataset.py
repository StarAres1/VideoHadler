import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = os.path.join(path, "train" if train else "test")
        self.transform = transform

        self.classes = []          # имена классов (строки)
        self.class_to_idx = {}      # отображение имени класса → индекс
        self.files = []             # список (путь, индекс_класса)

        # Обход поддиректорий
        for dir_name in sorted(os.listdir(self.path)):
            dir_path = os.path.join(self.path, dir_name)
            if not os.path.isdir(dir_path):
                continue

            # Добавляем новый класс, если его ещё нет
            if dir_name not in self.class_to_idx:
                idx = len(self.classes)
                self.class_to_idx[dir_name] = idx
                self.classes.append(dir_name)

            # Собираем все файлы изображений в этой директории
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                if os.path.isfile(file_path):
                    self.files.append((file_path, self.class_to_idx[dir_name]))

        # Создаём one-hot матрицу для всех классов
        num_classes = len(self.classes)
        self.targets = torch.eye(num_classes)

        self.length = len(self.files)

    def __getitem__(self, item):
        path_file, class_idx = self.files[item]
        img = Image.open(path_file)
        if self.transform:
            img = self.transform(img)
        t = self.targets[class_idx]   # one-hot вектор
        return img, t

    def __len__(self):
        return self.length


f = ImageDataset("../../dataset/dataset_ssim_contrasts")


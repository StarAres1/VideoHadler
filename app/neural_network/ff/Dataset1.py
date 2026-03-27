import os
from PIL import Image
import torch.utils.data as data


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
            for file_name in os.listdir(dir_path):
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
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
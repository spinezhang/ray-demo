import os
import numpy as np
from torch.utils.data import Dataset

from data_builder.image_data_builder import ImageDataBuilder


# pyarrow.dataset.InMemoryTest cannot serialize by pickle, so cannot use arrow_dataset.py.ArrowImageDataset
class MemoryTorchDataset(Dataset):
    def __init__(self, features, labels):
        self.images = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = ImageDataBuilder.image_transform(image)
        return image, label


class LocalTorchBuilder(ImageDataBuilder):
    def __init__(self, path):
        super().__init__(path)

    def to_dataset(self, is_train=True):
        images, labels = ImageDataBuilder.load_torchvision_data(self.data_path, is_train)
        features = np.array([np.frombuffer(image, dtype=np.uint8) for image in images])
        features = features.reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
        return MemoryTorchDataset(features, labels)

    def file_ready(self):
        return os.path.exists(self.data_path/"test_batch")

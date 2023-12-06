import os
import numpy as np
import ray
import pyarrow as pa
from PIL import Image
from pyarrow import RecordBatch
from torch.utils.data import Dataset

from image_data_builder import ImageDataBuilder


# pyarrow.dataset.InMemoryTest cannot serialize by pickle, so cannot use arrow_dataset.ArrowImageDataset
class MemoryTorchDataset(Dataset):
    def __init__(self, features, labels):
        self.images = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        image = ImageDataBuilder.transform(image)
        return image, label


class LocalBuilder(ImageDataBuilder):
    def __init__(self, path, ray_data=False):
        super().__init__(path, ray_data)

    def load_to_torch_dateset(self, is_train=True):
        images, labels = ImageDataBuilder.load_torchvision_data(self.data_path, is_train)
        features = np.array([np.frombuffer(image, dtype=np.uint8) for image in images])
        features = features.reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
        return MemoryTorchDataset(features, labels)

    def file_ready(self):
        return os.path.exists(self.data_path/"test_batch")

    def load_to_ray_dateset(self, is_train=True):
        images, labels = ImageDataBuilder.load_torchvision_data(self.data_path, is_train)
        pa_table = pa.table([images, labels], names=["image", "label"])
        return ray.data.from_arrow(pa_table)

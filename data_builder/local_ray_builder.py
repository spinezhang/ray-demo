import os
import ray
import pyarrow as pa

from data_builder.image_data_builder import ImageDataBuilder


class LocalRayBuilder(ImageDataBuilder):
    def __init__(self, path):
        super().__init__(path)

    def file_ready(self):
        return os.path.exists(os.path.join(self.data_path, "test_batch"))

    def to_dataset(self, is_train=True):
        images, labels = ImageDataBuilder.load_torchvision_data(self.data_path, is_train)
        pa_table = pa.table([images, labels], names=["image", "label"])
        return ray.data.from_arrow(pa_table)

import os
import deltalake
import ray
from deltalake import DeltaTable

from image_data_builder import ImageDataBuilder
from arrow_dataset import ArrowImageDataset


class DeltalakeImageBuilder(ImageDataBuilder):
    def __init__(self, path):
        super().__init__(path)

    def file_ready(self):
        return os.path.exists(self.table_file_path(False))

    def load_to_torch_dateset(self, is_train=True):
        delta_table = DeltaTable(self.table_file_path(is_train))
        return ArrowImageDataset(delta_table.to_pyarrow_dataset())

    def table_file_path(self, is_train):
        return f"{self.data_path}/{'train' if is_train else 'test'}.delta"

    def store_data(self, src_path, is_train=True):
        records = ImageDataBuilder.torchvision_files_to_pyarrow_records(src_path, is_train)
        deltalake.write_deltalake(self.table_file_path(is_train), records)

    def load_to_ray_dateset(self, is_train=True):
        delta_table = DeltaTable(self.table_file_path(is_train))
        dataset = ray.data.read_parquet([os.path.join(self.table_file_path(is_train), item) for item in delta_table.files()])
        return dataset

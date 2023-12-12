import os
import deltalake

from data_builder.image_data_builder import ImageDataBuilder


class DeltaLakeBuilder(ImageDataBuilder):
    def __init__(self, path):
        super().__init__(path)

    def file_ready(self):
        return os.path.exists(self.table_file_path(False))

    def table_file_path(self, is_train):
        return f"{self.data_path}/{'train' if is_train else 'test'}.delta"

    def store_data(self, src_path, is_train=True):
        records = ImageDataBuilder.torchvision_files_to_pyarrow_records(src_path, is_train)
        deltalake.write_deltalake(self.table_file_path(is_train), records)

    def to_dataset(self, is_train=True):
        pass

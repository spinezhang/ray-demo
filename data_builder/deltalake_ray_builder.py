import os
import ray
from deltalake import DeltaTable

from data_builder.deltalake_builder import DeltaLakeBuilder


class DeltaLakeRayBuilder(DeltaLakeBuilder):
    def __init__(self, path):
        super().__init__(path)

    def to_dataset(self, is_train=True):
        delta_table = DeltaTable(self.table_file_path(is_train))
        dataset = ray.data.read_parquet([os.path.join(self.table_file_path(is_train), item) for item in delta_table.files()])
        del delta_table
        return dataset
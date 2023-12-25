import os

from ray.data import read_parquet

from data_builder.deltalake_builder import DeltaLakeBuilder


class DeltaLakeRayBuilder(DeltaLakeBuilder):
    def __init__(self, path):
        super().__init__(path)

    def to_dataset(self, is_train=True):
        delta_table = self.read_table(is_train)
        dataset = read_parquet([os.path.join(self.table_file_path(is_train), item) for item in delta_table.files()])
        del delta_table
        return dataset

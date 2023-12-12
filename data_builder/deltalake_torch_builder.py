from deltalake import DeltaTable

from data_builder.deltalake_builder import DeltaLakeBuilder
from dataset.arrow_dataset_torch import ArrowDatasetTorch


class DeltaLakeTorchBuilder(DeltaLakeBuilder):
    def __init__(self, path):
        super().__init__(path)

    def to_dataset(self, is_train=True):
        delta_table = DeltaTable(self.table_file_path(is_train))
        dataset = ArrowDatasetTorch(delta_table.to_pyarrow_dataset())
        del delta_table
        return dataset

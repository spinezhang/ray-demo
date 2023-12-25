from data_builder.deltalake_builder import DeltaLakeBuilder
from dataset.arrow_dataset_torch import ArrowDatasetTorch


class DeltaLakeTorchBuilder(DeltaLakeBuilder):
    def __init__(self, path):
        super().__init__(path)

    def to_dataset(self, is_train=True):
        table = self.read_table(is_train)
        dataset = ArrowDatasetTorch(table.to_pyarrow_dataset())
        del table
        return dataset

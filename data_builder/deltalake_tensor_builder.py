from tensorflow.python import tf2
import tensorflow as tf

from data_builder.deltalake_builder import DeltaLakeBuilder
from dataset.arrow_dataset_tensor import ArrowDatasetTensor

tf2.enable()


class DeltaLakeTensorBuilder(DeltaLakeBuilder):
    def __init__(self, path):
        super().__init__(path)

    def to_dataset(self, is_train=True):
        table = self.read_table(is_train)
        arrow_dataset = ArrowDatasetTensor(table.to_pyarrow_dataset())
        output_types = (tf.float32, tf.int64)
        return tf.data.Dataset.from_generator(arrow_dataset, output_types=output_types)

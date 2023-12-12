from deltalake import DeltaTable
from tensorflow.python import tf2
import tensorflow as tf

from data_builder.deltalake_builder import DeltaLakeBuilder
from dataset.arrow_dataset_tensor import ArrowDatasetTensor

tf2.enable()


class DeltaLakeTensorBuilder(DeltaLakeBuilder):
    def __init__(self, path):
        super().__init__(path)

    def to_dataset(self, is_train=True):
        pa_dataset = DeltaTable(self.table_file_path(is_train)).to_pyarrow_dataset()
        arrow_dataset = ArrowDatasetTensor(pa_dataset)
        output_types = (tf.float32, tf.int64)
        return tf.data.Dataset.from_generator(arrow_dataset, output_types=output_types)
        # output_signature = (tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int64))
        # return tf.data.Dataset.from_generator(arrow_dataset, output_signature=output_signature)

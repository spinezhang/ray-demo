# import numpy as np
# from petastorm import TransformSpec
# from petastorm.spark import make_spark_converter, SparkDatasetConverter

from data_builder.deltalake_tensor_builder import DeltaLakeTensorBuilder
from data_builder.deltaspark_builder import DeltaSparkBuilder
# from data_builder.image_data_builder import ImageDataBuilder


class DeltaSparkTensorBuilder(DeltaSparkBuilder, DeltaLakeTensorBuilder):
    def __init__(self, path):
        DeltaSparkBuilder.__init__(self, path)
        DeltaLakeTensorBuilder.__init__(self, path)

    # def to_dataset(self, is_train=True):
    #     self.spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, "file:///tmp/petastorm")
    #     df = self.spark.read.format("delta").load(self.table_file_path(is_train))
    #     converter = make_spark_converter(df)
    #     transform_spec_fn = TransformSpec(
    #         transform_row,
    #         edit_fields=[('features', np.float32, (32, 32, 3), False), ('label', np.int64, (), False)],
    #         selected_fields=['features', 'label']
    #     )
    #     with converter.make_tf_dataset(transform_spec=transform_spec_fn, batch_size=32) as tf_dataset:
    #         result = tf_dataset.map(lambda x: (x.features, x.label))
    #     return result

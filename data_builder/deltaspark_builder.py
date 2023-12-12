import pyspark
from delta import configure_spark_with_delta_pip
from pyspark.sql.types import StructType, StructField, BinaryType, LongType

from data_builder.deltalake_builder import DeltaLakeBuilder
from data_builder.image_data_builder import ImageDataBuilder


class DeltaSparkBuilder(DeltaLakeBuilder):
    def __init__(self, path):
        super().__init__(path)

        builder = (
            pyspark.sql.SparkSession.builder.appName("deltatorch-demo")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        )
        self.spark = configure_spark_with_delta_pip(builder).getOrCreate()

    def store_data(self, src_path, is_train=True):
        records = ImageDataBuilder.torchvision_files_to_pyarrow_records(src_path, is_train).to_pandas()
        df = self.spark.createDataFrame(records, StructType([StructField("id", LongType()), StructField("image", BinaryType()), StructField("label", LongType())]))
        df.write.format("delta").mode("overwrite").save(path=self.table_file_path(is_train))

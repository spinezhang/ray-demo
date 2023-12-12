from data_builder.deltalake_ray_builder import DeltaLakeRayBuilder
from data_builder.deltaspark_builder import DeltaSparkBuilder


class DeltaSparkRayBuilder(DeltaSparkBuilder, DeltaLakeRayBuilder):
    def __init__(self, path):
        DeltaSparkBuilder.__init__(self, path)
        DeltaLakeRayBuilder.__init__(self, path)

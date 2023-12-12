from data_builder.deltalake_torch_builder import DeltaLakeTorchBuilder
from data_builder.deltaspark_builder import DeltaSparkBuilder


class DeltaSparkTorchBuilder(DeltaSparkBuilder, DeltaLakeTorchBuilder):
    def __init__(self, path):
        DeltaSparkBuilder.__init__(self, path)
        DeltaLakeTorchBuilder.__init__(self, path)

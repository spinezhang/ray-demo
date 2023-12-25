import json
import os
import deltalake
from deltalake import DeltaTable

from data_builder.image_data_builder import ImageDataBuilder


class DeltaLakeBuilder(ImageDataBuilder):
    def __init__(self, path):
        super().__init__(path)

    def file_ready(self):
        return os.path.exists(self.table_file_path(False))

    def table_file_path(self, is_train):
        return f"{self.data_path}/{'train' if is_train else 'test'}.delta"

    def read_table(self, is_train):
        delta_table = DeltaTable(self.table_file_path(is_train))
        self.update_metadata(delta_table.metadata().description)
        return delta_table

    def update_metadata(self, metadata):
        if self._metadata is None:
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            self._metadata = metadata

    def store_data(self, data, metadata, is_train=True, schema=None):
        self.update_metadata(metadata)
        description = json.dumps(metadata)
        deltalake.write_deltalake(self.table_file_path(is_train), data, schema=schema, description=description, engine='rust')  # engine='rust', 'pyarrow'

    def save_image(self, image, label, is_train=True):
        pass

    def to_dataset(self, is_train=True):
        pass

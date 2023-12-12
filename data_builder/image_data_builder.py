import os
import pickle
from abc import abstractmethod

import numpy as np
from pyarrow import RecordBatch
import pyarrow as pa


class ImageDataBuilder:
    def __init__(self, path):
        self.data_path = path

    @staticmethod
    def read_torchvision_files(src_path, file_list):
        features = []
        labels = []
        for file_name in file_list:
            file_path = os.path.join(src_path, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                features.extend(map(lambda x: x.tobytes(), entry['data']))
                if 'labels' in entry:
                    labels.extend(entry['labels'])
                else:
                    labels.extend(entry['fine_labels'])
                f.close()
        return features, labels

    @staticmethod
    def load_torchvision_meta(src_path):
        path = os.path.join(src_path, 'batches.meta')
        metadata = {}
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            classes = data['label_names']
            metadata = {'mode': 'RGB', 'size': '32, 32', 'classes': classes}
        return metadata

    @staticmethod
    def load_torchvision_data(path, is_train=True):
        if is_train:
            file_list = [f"data_batch_{i + 1}" for i in range(5)]
        else:
            file_list = ["test_batch"]
        return ImageDataBuilder.read_torchvision_files(path, file_list)

    @staticmethod
    def torchvision_files_to_pyarrow_records(file_path, is_train):
        images, labels = ImageDataBuilder.load_torchvision_data(file_path, is_train)
        ids = list(range(len(labels)))
        schema = pa.schema([pa.field('id', pa.int64()), pa.field('image', pa.binary()), pa.field('label', pa.int64())], metadata={'mode': 'RGB', 'size': '32, 32'})
        return RecordBatch.from_arrays([ids, images, labels], schema)

    @staticmethod
    def image_transform(pic):
        image = np.array([np.frombuffer(pic, dtype=np.uint8)]).astype(np.float32) / 255.0
        image = image.reshape(3, 32, 32)
        return image

    @abstractmethod
    def file_ready(self):
        pass

    @abstractmethod
    def to_dataset(self, is_train=True):
        pass

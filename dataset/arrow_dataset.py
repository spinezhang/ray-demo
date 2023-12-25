import random
import numpy as np
from pyarrow import compute

from data_builder.image_data_builder import ImageDataBuilder


class ArrowDataset:
    def __init__(self, pa_dataset, shuffle=False):
        super().__init__()
        self.pa_dataset = pa_dataset
        self.shuffle = shuffle

    def count(self):
        return self.pa_dataset.count_rows()

    def read_and_decode(self, iter_start, iter_end):
        filter_id = None
        if iter_end > 0 and iter_start >= 0:
            filter_id = (compute.field('id') >= compute.scalar(iter_start)) & (compute.field('id') < compute.scalar(iter_end))

        scanner = self.pa_dataset.scanner(columns=['image', 'label'], filter=filter_id)
        for rb in scanner.to_reader():
            num_rows = rb.num_rows
            indexes = list(range(num_rows))
            if self.shuffle:
                random.shuffle(indexes)
            for i in indexes:
                item = rb.slice(offset=i, length=1).to_pylist()[0]
                image = ImageDataBuilder.image_transform(item['image'])
                row = image, np.array(item['label'])  # torch
                yield row

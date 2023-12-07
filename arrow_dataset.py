import math
import random
import torch
from torch.utils.data import IterableDataset, get_worker_info
from pyarrow import compute

from image_data_builder import ImageDataBuilder


class ArrowImageDataset(IterableDataset):
    def __getitem__(self, index):
        pass

    def __init__(self, pa_dataset, shuffle=True):
        super().__init__()
        self.pa_dataset = pa_dataset
        self.shuffle = shuffle
        self.start = 0
        self.end = self.pa_dataset.count_rows()
        self.num_ranks = 1
        self.rank = 1
        self.init_boundaries()

    def __len__(self):
        return int(self.end - self.start)

    def __iter__(self):
        filter_id = None
        iter_start, iter_end = self.calc_chunk_boundaries_for_current_worker()
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
                row = (image, item['label'])
                yield row

    def init_boundaries(self):
        if torch.distributed.is_initialized():
            self.num_ranks = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
            self.start, self.end = self.calc_boundaries(self.start, self.end, self.rank, self.num_ranks)

    def calc_chunk_boundaries_for_current_worker(self):
        worker_info = get_worker_info()
        if worker_info is None:
            return self.start, self.end
        else:
            iter_start, iter_end = self.calc_boundaries(self.start, self.end, worker_info.id, worker_info.num_workers)
        return iter_start, iter_end

    @staticmethod
    def calc_boundaries(start, end, rank, num_ranks):
        per_worker_data_count = int(math.ceil((end - start) / float(num_ranks)))
        new_start = start + rank * per_worker_data_count
        new_end = min(start + (rank + 1) * per_worker_data_count, end)
        return new_start, new_end


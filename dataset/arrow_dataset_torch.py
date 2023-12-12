import math
import random
import torch
from torch.utils.data import IterableDataset, get_worker_info

from dataset.arrow_dataset import ArrowDataset


class ArrowDatasetTorch(IterableDataset):
    def __getitem__(self, index):
        pass

    def __init__(self, pa_dataset, shuffle=True):
        super().__init__()
        self.arrow_dataset = ArrowDataset(pa_dataset, shuffle)
        self.start = 0
        self.end = self.arrow_dataset.count()
        self.num_ranks = 1
        self.rank = 1
        self.init_boundaries()

    def __len__(self):
        return int(self.end - self.start)

    def __iter__(self):
        iter_start, iter_end = self.calc_chunk_boundaries_for_current_worker()
        return self.arrow_dataset.read_and_decode(iter_start, iter_end)

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

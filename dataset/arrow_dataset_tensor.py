from dataset.arrow_dataset import ArrowDataset


class ArrowDatasetTensor:
    def __init__(self, pa_dataset, shuffle=False):
        self.arrow_dataset = ArrowDataset(pa_dataset, shuffle)
        self.start = 0
        self.end = self.arrow_dataset.count()

    def __call__(self):
        return self.arrow_dataset.read_and_decode(0,  self.end)

import torch
from torch.utils.data import DataLoader

from deltaspark_builder import DeltaSparkBuilder
from deltalake_builder import DeltalakeImageBuilder
from local_builder import LocalBuilder


def verify_data(data_loader, once=True):
    count = 0
    for i, item in enumerate(data_loader):
        assert isinstance(item[0], torch.Tensor)
        assert item[0].shape == torch.Size([1, 3, 32, 32])
        count = i
        if once:
            break
    return count+1


def test_local_data():
    image_data = LocalBuilder("../data/cifar-10-batches-py")
    test_data = image_data.load_to_torch_dateset(is_train=False)
    data_loader = DataLoader(test_data)
    count = verify_data(data_loader, False)
    assert count == 10000


def data_lake_verify(builder, path, use_raydata=False):
    image_data = builder(path)
    if not image_data.file_ready():
        image_data.store_data("../data/cifar-10-batches-py", is_train=True)
        image_data.store_data("../data/cifar-10-batches-py", is_train=False)
    if use_raydata:
        test_data = image_data.load_to_ray_dateset(is_train=False)
    else:
        test_data = image_data.load_to_torch_dateset(is_train=False)
    data_loader = DataLoader(test_data)
    count = verify_data(data_loader, False)
    assert count == 10000


def test_delta_spark():
    data_lake_verify(DeltaSparkBuilder, "../data/delta_spark")


def test_delta_lake():
    data_lake_verify(DeltalakeImageBuilder, "../data/delta_lake")

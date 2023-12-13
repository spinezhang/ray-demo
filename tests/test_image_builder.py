import os

import torch
from torch.utils.data import DataLoader
import tensorflow as tf

from data_builder.deltalake_tensor_builder import DeltaLakeTensorBuilder
from data_builder.deltaspark_builder import DeltaSparkBuilder
from data_builder.deltalake_torch_builder import DeltaLakeTorchBuilder
from data_builder.deltaspark_torch_builder import DeltaSparkTorchBuilder
from data_builder.local_torch_builder import LocalTorchBuilder
from image_train_tensor import _fixup_shape


def verify_torch_data(data_loader, once=True):
    count = 0
    for i, item in enumerate(data_loader):
        assert isinstance(item[0], torch.Tensor)
        assert item[0].shape == torch.Size([1, 3, 32, 32])
        count = i
        if once:
            break
    return count+1


def test_local_torch():
    image_data = LocalTorchBuilder("../data/cifar-10-batches-py")
    test_data = image_data.to_dataset(is_train=False)
    data_loader = DataLoader(test_data)
    count = verify_torch_data(data_loader, False)
    assert count == 10000


def data_lake_torch_verify(builder, path, use_raydata=False):
    image_data = builder(path)
    if not image_data.file_ready():
        image_data.store_data("../data/cifar-10-batches-py", is_train=True)
        image_data.store_data("../data/cifar-10-batches-py", is_train=False)
    if use_raydata:
        test_data = image_data.load_to_ray_dateset(is_train=False)
    else:
        test_data = image_data.load_to_torch_dateset(is_train=False)
    data_loader = DataLoader(test_data)
    count = verify_torch_data(data_loader, False)
    assert count == 10000


def test_delta_spark_torch():
    data_lake_torch_verify(DeltaSparkTorchBuilder, "../data/delta_spark")


def test_delta_lake_torch():
    data_lake_torch_verify(DeltaLakeTorchBuilder, "../data/delta_lake")


def test_local_tensor(work_path):
    image_data = LocalTorchBuilder("../data/cifar-10-batches-py")
    test_data = image_data.to_dataset(is_train=False)
    # test_data = test_data.map(ImageDataBuilder.tensor_image_transform)
    for i, item in enumerate(test_data):
        if i == 10:
            break


def test_delta_lake_tensor(work_path):
    tf.data.experimental.enable_debug_mode()
    image_data = DeltaLakeTensorBuilder(os.path.join(work_path, 'data/delta_lake'))
    test_data = image_data.to_dataset(is_train=False).map(_fixup_shape)
    # test_data.map(ImageDataBuilder.tensor_image_transform)
    for i, item in enumerate(test_data):
        print(f"label:{item[1].numpy()}, len:{len(item[1].numpy())}, shape:{item[1].shape}")
        if i == 3:
            break
        # assert image[0].shape == [1, 3, 32, 32]

import os

from data_builder.deltalake_torch_builder import DeltaLakeTorchBuilder
from model.cnn_torch import CnnTorchModel
from image_train_torch import ImageTorchInferenceSingle


def prepare_model_and_data_builder(InferenceClass, model_path, data_path, use_gpu=False):
    net = CnnTorchModel(num_classes=10)
    # model = AlexNet(num_classes=10)
    inference = InferenceClass(net, model_path, use_gpu)
    image_data = DeltaLakeTorchBuilder(data_path)
    return inference, image_data


def image_torch_inference_single(model_path, data_path, expected_acc):
    inference, image_data = prepare_model_and_data_builder(ImageTorchInferenceSingle, model_path, data_path)
    test_dataset = image_data.to_dataset(is_train=False)
    test_acc = inference.batch_predict(test_dataset)
    print(f"test_acc:{test_acc}")
    assert test_acc > expected_acc


def test_image_torch_inference_single(work_path):
    image_torch_inference_single(os.path.join(work_path, 'cifar10_torch_single.model'), os.path.join(work_path, 'data/delta_lake'), 0.74)


def test_torch_single_inference_with_ray_model(work_path):
    image_torch_inference_single(os.path.join(work_path, 'cifar10_ray_alex.model'), os.path.join(work_path, 'data/delta_lake'), 0.74)

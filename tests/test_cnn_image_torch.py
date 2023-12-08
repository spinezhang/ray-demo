from cnn_image_torch import ImageTorchInferenceSingle, ImageCnnTorch
from deltalake_builder import DeltalakeImageBuilder


def image_torch_inference_single(path, expected_acc):
    net = ImageCnnTorch(num_classes=10)
    inference = ImageTorchInferenceSingle(net, path)
    image_data = DeltalakeImageBuilder("../data/delta_lake")
    test_dataset = image_data.load_to_torch_dateset(is_train=False)
    test_acc = inference.test_dataset(test_dataset)
    assert test_acc > expected_acc


def test_image_torch_inference_single():
    image_torch_inference_single("../cifar10_torch_single.model", 0.74)


def test_torch_single_inference_with_ray_model():
    image_torch_inference_single("../cifar10_torch_ray.model", 0.74)

from cnn_image_torch import ImageTorchInferenceSingle, ImageCnnTorch
from deltalake_builder import DeltalakeImageBuilder


def test_image_torch_inference_single():
    net = ImageCnnTorch(num_classes=10)
    inference = ImageTorchInferenceSingle(net, "../cifar10_torch_single.model")
    image_data = DeltalakeImageBuilder("../data/delta_lake")
    test_dataset = image_data.load_to_torch_dateset(is_train=False)
    test_acc = inference.test_dataset(test_dataset)
    assert test_acc > 0.74


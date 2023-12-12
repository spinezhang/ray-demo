import os

from image_train_raytorch import ImageTorchInferenceRay
from tests.test_cnn_image_torch import prepare_model_and_data_builder

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "cifar10_torch_ray.model")
    data_path = os.path.join(current_dir, "data/delta_lake")
    inference, image_data = prepare_model_and_data_builder(ImageTorchInferenceRay, model_path, data_path)
    test_dataset = image_data.load_to_ray_dateset(is_train=False)
    test_acc = inference.batch_predict(test_dataset, batch_size=200, num_workers=8)
    print(f"inference:{test_acc}")

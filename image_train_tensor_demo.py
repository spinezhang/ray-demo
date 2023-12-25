"""
Separated from image train demo, for on of my conda python 3.10 environments has protobuf probelm.
the Ray cannot work with pytorch, tensorflow and ray simultaneously. (Fixed by removing the unknown dynamic library
in protobuf package).
At the sametime, the data builders are refactored to pytorch, tensorflow and ray, so this package can work with either
pytorch or tensorflow, do not need to install both, this is useful for Docker and cluster environments.
"""

import os
import traceback

from data_builder.deltalake_ray_builder import DeltaLakeRayBuilder
from data_builder.deltalake_tensor_builder import DeltaLakeTensorBuilder
# from data_builder.deltaspark_tensor_builder import DeltaSparkTensorBuilder
from image_train_raytensor import build_and_train_ray_tf
from image_train_tensor import ImageTrainerTFSingle
from datetime import datetime

from model.cnn_tensor import CnnTensorModel

if __name__ == "__main__":
    start_time = datetime.now()
    ray = 1
    storage = "deltalake"

    in_k8s = os.getenv("KUBERNETES_SERVICE_PORT")
    if in_k8s:
        data_dir = '/data0/cifar10'
    else:
        data_dir = os.path.dirname(os.path.abspath(__file__))+'/../cifar10'
    original_file_path = os.path.join(data_dir, "cifar-10-batches-py")
    # if storage == "deltaspark":
    #     image_data = DeltaSparkTensorBuilder(os.path.join(current_dir, "data/delta_spark"))
    # else:
    if ray == 1:
        data_builder = DeltaLakeRayBuilder(os.path.join(data_dir, "delta_lake"))
    else:
        data_builder = DeltaLakeTensorBuilder(os.path.join(data_dir, "delta_lake"))

    train_data = data_builder.to_dataset(is_train=True)
    test_data = data_builder.to_dataset(is_train=False)

    try:
        train_config = {"use_gpu": True, "num_epochs": 3, "batch_size": 50, "num_workers": 4}
        if ray == 1:
            train_config['use_gpu'] = False  # GPU distribution not working on Mac
            build_and_train_ray_tf(CnnTensorModel, train_data, test_data, train_config)
        else:
            model = CnnTensorModel(num_classes=10)
            ImageTrainerTFSingle.build_and_train(model, train_data, test_data, train_config)

        end_time = datetime.now()
        print(f"test_image used {round((end_time - start_time).total_seconds(), 3)}s")
    except Exception as e:
        print(e)
        traceback.print_exc()
        exit(1)

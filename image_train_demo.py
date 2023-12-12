import getopt
import os
import sys
import traceback

from data_builder.deltalake_ray_builder import DeltaLakeRayBuilder
from data_builder.deltalake_tensor_builder import DeltaLakeTensorBuilder
from data_builder.deltalake_torch_builder import DeltaLakeTorchBuilder
from data_builder.deltaspark_ray_builder import DeltaSparkRayBuilder
from data_builder.deltaspark_tensor_builder import DeltaSparkTensorBuilder
from data_builder.deltaspark_torch_builder import DeltaSparkTorchBuilder
from data_builder.local_ray_builder import LocalRayBuilder
from image_train_raytensor import ImageTrainerTFRay
from image_train_raytorch import ImageTrainerTorchRay
from image_cnn_tensor import ImageTrainerTFSingle
from image_train_torch import ImageTrainerTorchSingle
from data_builder.local_torch_builder import LocalTorchBuilder
from datetime import datetime


def parse_args(argv):
    storage_f = "deltalake"
    framework_f = "torch"
    ray_f = "1"

    short_opts = "h:s:f:r:"
    long_opts = ["help", "storage", "framework", "ray"]
    opts, args = getopt.getopt(argv, short_opts, long_opts)
    for option, value in opts:
        if option in ("-h", "--help"):
            print("Usage python image_train_demo.py -s=<local|deltaspark|deltalake> -f=<torch|tf> -r=<1(ray cluster)|0(single)>")
            exit(0)
        elif option in ("-s", "--storage"):
            storage_f = value
        elif option in ("-f", "--framework"):
            framework_f = value
        elif option in ("-r", "--ray"):
            ray_f = value

    return storage_f, framework_f, ray_f


def get_data_builder(storage, framework, ray):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    original_file_path = os.path.join(current_dir, "data/cifar-10-batches-py")
    if storage == "deltalake":
        if ray == "1":
            data_builder = DeltaLakeRayBuilder(os.path.join(current_dir, "data/delta_lake"))
        elif framework == 'torch':
            data_builder = DeltaLakeTorchBuilder(os.path.join(current_dir, "data/delta_lake"))
        else:
            data_builder = DeltaLakeTensorBuilder(os.path.join(current_dir, "data/delta_lake"))
    elif storage == "deltaspark":
        if ray == "1":
            data_builder = DeltaSparkRayBuilder(os.path.join(current_dir, "data/delta_spark"))
        elif framework == 'torch':
            data_builder = DeltaSparkTorchBuilder(os.path.join(current_dir, "data/delta_spark"))
        else:
            data_builder = DeltaSparkTensorBuilder(os.path.join(current_dir, "data/delta_spark"))
    else:
        if ray == "1":
            data_builder = LocalRayBuilder(original_file_path)
        elif framework == 'torch':
            data_builder = LocalTorchBuilder(original_file_path)
        # else:
        #     data_builder = DeltaSparkTensorBuilder(current_dir, "data/delta_spark")
    if not isinstance(data_builder, LocalTorchBuilder) and not data_builder.file_ready():
        data_builder.store_data(original_file_path, is_train=True)
        data_builder.store_data(original_file_path, is_train=False)

    train_data = data_builder.to_dataset(is_train=True)
    test_data = data_builder.to_dataset(is_train=False)

    return train_data, test_data


if __name__ == "__main__":
    storage, framework, ray = parse_args(sys.argv[1:])
    print(f"args: storage:{storage}, framework:{framework}, ray:{ray}")
    start_time = datetime.now()

    train_data, test_data = get_data_builder(storage, framework, ray)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        train_config = {"num_classes": 10, "use_gpu": True, "num_epochs": 2, "batch_size": 50, "num_workers": 6}
        if framework == "torch":
            if ray == "1":
                train_config['use_gpu'] = False  # GPU distribution not working on Mac
                train_config['work_dir'] = current_dir
                ImageTrainerTorchRay.build_and_train(train_data, test_data, train_config)
            else:
                ImageTrainerTorchSingle.build_and_train(train_data, test_data, train_config)
        else:
            if ray == "1":
                train_config['use_gpu'] = False  # GPU distribution not working on Mac
                train_config['work_dir'] = current_dir
                ImageTrainerTFRay.build_and_train(train_data, test_data, train_config)
            else:
                ImageTrainerTFSingle.build_and_train(train_data, test_data, train_config)

        end_time = datetime.now()
        print(f"test_image used {round((end_time - start_time).total_seconds(), 3)}s")
    except Exception as e:
        print(e)
        traceback.print_exc()
        exit(1)

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
from data_builder.image_data_builder import ImageDataBuilder
from data_builder.local_ray_builder import LocalRayBuilder
from model.cnn_tensor import CnnTensorModel
from model.cnn_torch import CnnTorchModel
from image_train_raytensor import build_and_train_ray_tf
from image_train_raytorch import ImageTrainerTorchRay
from image_train_tensor import ImageTrainerTFSingle
from image_train_torch import ImageTrainerTorchSingle
from data_builder.local_torch_builder import LocalTorchBuilder
from datetime import datetime


def parse_args(argv):
    storage_f = "local"
    framework_f = "torch"
    data_path = '/data0/cifar10'
    ray_f = "0"

    short_opts = "h:d:s:f:r:"
    long_opts = ["help", "data-path", "storage", "framework", "ray"]
    opts, args = getopt.getopt(argv, short_opts, long_opts)
    for option, value in opts:
        if option in ("-h", "--help"):
            print("Usage python image_train_demo.py -d=[data-path] -s=<local|deltaspark|deltalake> -f=<torch|tf> -r=<1(ray cluster)|0(single)>")
            exit(0)
        elif option in ("-d", "--data-path"):
            data_path = value
        elif option in ("-s", "--storage"):
            storage_f = value
        elif option in ("-f", "--framework"):
            framework_f = value
        elif option in ("-r", "--ray"):
            ray_f = value

    return data_path, storage_f, framework_f, ray_f


def get_data_builder(data_dir, storage, framework, ray):
    original_file_path = os.path.join(data_dir, "cifar-10-batches-py")
    if storage == "deltalake":
        if ray == "1":
            data_builder = DeltaLakeRayBuilder(os.path.join(data_dir, "delta_lake"))
        elif framework == 'torch':
            data_builder = DeltaLakeTorchBuilder(os.path.join(data_dir, "delta_lake"))
        else:
            data_builder = DeltaLakeTensorBuilder(os.path.join(data_dir, "delta_lake"))
    elif storage == "deltaspark":
        if ray == "1":
            data_builder = DeltaSparkRayBuilder(os.path.join(data_dir, "delta_spark"))
        elif framework == 'torch':
            data_builder = DeltaSparkTorchBuilder(os.path.join(data_dir, "delta_spark"))
        else:
            data_builder = DeltaSparkTensorBuilder(os.path.join(data_dir, "delta_spark"))
    else:
        if ray == "1":
            data_builder = LocalRayBuilder(original_file_path)
        elif framework == 'torch':
            data_builder = LocalTorchBuilder(original_file_path)
        # else:
        #     data_builder = DeltaSparkTensorBuilder(data_dir, "data/delta_spark")
    if not isinstance(data_builder, LocalTorchBuilder) and not data_builder.file_ready():
        train_records = ImageDataBuilder.torchvision_files_to_pyarrow_records(original_file_path, is_train=True)
        image_info = ImageDataBuilder.load_torchvision_meta(original_file_path)
        data_builder.store_data(train_records, image_info, is_train=True)

        test_records = ImageDataBuilder.torchvision_files_to_pyarrow_records(original_file_path, is_train=True)
        data_builder.store_data(test_records, image_info, is_train=False)

    train_data_dataset = data_builder.to_dataset(is_train=True)
    test_dataset = data_builder.to_dataset(is_train=False)
    metadata = data_builder.metadata()

    return train_data_dataset, test_dataset, metadata


if __name__ == "__main__":
    data_dir, storage, framework, ray = parse_args(sys.argv[1:])
    print(f"data-path: {data_dir}, args: storage:{storage}, framework:{framework}, ray:{ray}")
    start_time = datetime.now()

    train_data, test_data, metadata = get_data_builder(data_dir, storage, framework, ray)
    print(metadata)

    try:
        train_config = {"use_gpu": True, "num_epochs": 2, "batch_size": 50, "num_workers": 6}
        if ray == "1":
            train_config['use_gpu'] = False  # GPU distribution not working on Mac
            train_config['work_dir'] = '/tmp/ray'

        if framework == 'torch':
            model = CnnTorchModel(num_classes=10)
            # model = AlexNet(num_classes=10)
        elif ray == "1":
            # ray tensorflow does not support pass model instance, for the 'name' in Sequential cannot be pickled.
            # the model must be instanced in worker entry and after the strategy is created, so here pass the class type.
            model = CnnTensorModel
        else:
            model = CnnTensorModel(num_classes=10)

        if framework == "torch":
            if ray == "1":
                ImageTrainerTorchRay.build_and_train(model, train_data, test_data, train_config)
            else:
                ImageTrainerTorchSingle.build_and_train(model, train_data, test_data, train_config)
        else:
            if ray == "1":
                build_and_train_ray_tf(model, train_data, test_data, train_config)
            else:
                ImageTrainerTFSingle.build_and_train(model, train_data, test_data, train_config)

        end_time = datetime.now()
        print(f"test_image used {round((end_time - start_time).total_seconds(), 3)}s")
    except Exception as e:
        print(e)
        traceback.print_exc()
        exit(1)

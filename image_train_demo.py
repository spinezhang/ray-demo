import getopt
import os
import sys
import traceback
from cnn_image_raytorch import ImageTrainerTorchRay
from cnn_image_tensor import ImageTrainerTFSingle
from cnn_image_torch import ImageTrainerTorchSingle
from deltalake_builder import DeltalakeImageBuilder
from deltaspark_builder import DeltaSparkBuilder
from local_builder import LocalBuilder
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


if __name__ == "__main__":
    storage, framework, ray = parse_args(sys.argv[1:])
    print(f"args: storage:{storage}, framework:{framework}, ray:{ray}")
    start_time = datetime.now()

    if ray == "1":
        ray_data = True
    else:
        ray_data = False

    current_dir = os.path.dirname(os.path.abspath(__file__))
    original_file_path = os.path.join(current_dir, "data/cifar-10-batches-py")
    if storage == "deltaspark":
        image_data = DeltaSparkBuilder(os.path.join(current_dir, "data/delta_spark"))
    elif storage == "deltalake":
        image_data = DeltalakeImageBuilder(os.path.join(current_dir, "data/delta_lake"))
    else:
        image_data = LocalBuilder(original_file_path)

    if not isinstance(image_data, LocalBuilder) and not image_data.file_ready():
        image_data.store_data(original_file_path, is_train=True)
        image_data.store_data(original_file_path, is_train=False)

    try:
        train_config = {"num_classes": 10, "use_gpu": True, "num_epochs": 5, "batch_size": 50, "num_workers": 6}
        if framework == "torch":
            if ray == "1":
                train_config['use_gpu'] = False  # GPU distribution not working on Mac
                train_config['work_dir'] = current_dir
                ImageTrainerTorchRay.build_and_train(image_data, train_config)
            else:
                ImageTrainerTorchSingle.build_and_train(image_data, train_config)
        else:
            ImageTrainerTFSingle.build_and_train(image_data, train_config)

        end_time = datetime.now()
        print(f"test_image used {round((end_time - start_time).total_seconds(), 3)}s")
    except Exception as e:
        print(e)
        traceback.print_exc()
        exit(1)

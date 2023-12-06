import getopt
import sys
import traceback
from cnn_image_raytorch import ImageTrainerTorchRay
from cnn_image_torch import ImageTrainerTorchSingle
from deltalake_builder import DeltalakeImageBuilder
from deltaspark_builder import DeltaSparkBuilder
from local_builder import LocalBuilder
from datetime import datetime


def parse_args(argv):
    storage_f = "deltaspark"
    framework_f = "pytorch"
    ray_f = 1

    short_opts = "h:s:f:r:"
    long_opts = ["help", "storage", "framework", "ray"]
    opts, args = getopt.getopt(argv, short_opts, long_opts)
    for option, value in opts:
        if option in ("-h", "--help"):
            print("Usage python image_train_demo.py -s=<local|deltaspark|deltalake> -f=<pytorch|tensorflow> -r=<1(ray cluster)|0(single)>")
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
    start_time = datetime.now()

    if ray == 1:
        ray_data = True
    else:
        ray_data = False

    if storage == "deltaspark":
        image_data = DeltaSparkBuilder("./data/delta_spark", ray_data)
    elif storage == "deltalake":
        image_data = DeltalakeImageBuilder("./data/delta_lake", ray_data)
    else:
        image_data = LocalBuilder("./data/cifar-10-batches-py", ray_data=False)

    if not isinstance(image_data, LocalBuilder) and not image_data.file_ready():
        image_data.store_data("./data/cifar-10-batches-py", is_train=True)
        image_data.store_data("./data/cifar-10-batches-py", is_train=False)

    try:
        train_config = {"num_classes": 10, "use_gpu": False, "num_epochs": 2, "batch_size": 64, "num_workers": 8}
        if ray == 1:
            ImageTrainerTorchRay.build_and_train(image_data, train_config)
        else:
            ImageTrainerTorchSingle.build_and_train(image_data, train_config)

        end_time = datetime.now()
        print(f"test_image used {round((end_time - start_time).total_seconds(), 3)}s")
    except Exception as e:
        print(e)
        traceback.print_exc()
        exit(1)

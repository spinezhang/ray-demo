import os

import pickle
import numpy as np
import ray
import pyarrow as pa
from ray import train
from ray.air import ScalingConfig, RunConfig, CheckpointConfig
from torchvision import datasets

import tensorflow as tf
from ray.train.tensorflow import TensorflowTrainer, prepare_dataset_shard
from keras import layers, Sequential


class CnnTensorModel(Sequential):
    def __init__(self, num_classes=10):
        super().__init__([
            layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
            layers.BatchNormalization(),

            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dropout(0.2),

            # Hidden layer
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.2),

            layers.Dense(num_classes, activation='softmax')
        ], name='cnn')

        self.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


def torchvision_ray_dataset(path, is_train=True):
    if is_train:
        file_list = [f"data_batch_{i + 1}" for i in range(5)]
    else:
        file_list = ["test_batch"]

    images = []
    labels = []
    for file_name in file_list:
        file_path = os.path.join(path, file_name)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            images.extend(map(lambda x: x.tobytes(), entry['data']))
            if 'labels' in entry:
                labels.extend(entry['labels'])
            f.close()
    pa_table = pa.table([images, labels], names=["image", "label"])
    return ray.data.from_arrow(pa_table)


def to_tf_dataset(dataset, batch_size):
    def image_transform(pic):
        image = np.array([np.frombuffer(pic, dtype=np.uint8)]).astype(np.float32) / 255.0
        image = image.reshape(32, 32, 3)
        return image

    def to_tensor_iterator():
        # !!! FATAL, iter_tf_batches has critical bug, somtimes it will change the bytes array size, this cause
        # ImageDataBuilder.image_transform() transpose array failed.

        for batch in dataset.iter_tf_batches(batch_size=batch_size):
        # for batch in dataset.iter_batches(batch_size=batch_size):
            images = tf.convert_to_tensor(list(map(lambda x: image_transform(x.numpy()), batch['image'])))
            # images = tf.convert_to_tensor(list(map(lambda x: image_transform(x), batch['image'])))
            images.set_shape((None, 32, 32, 3))
            label = tf.cast(tf.expand_dims(batch["label"], 1), tf.int64)
            yield images, label

    output_signature = (tf.TensorSpec(shape=(None, 32, 32, 3), dtype=tf.float32), tf.TensorSpec(shape=(None, 1), dtype=tf.int64))
    tf_dataset = tf.data.Dataset.from_generator(to_tensor_iterator, output_signature=output_signature)
    return prepare_dataset_shard(tf_dataset)


def train_loop_func(config):
    os.environ.pop('TF_CONFIG', None)
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    results = {}
    with strategy.scope():
        model = CnnTensorModel(num_classes=10)
        train_dataset = train.get_dataset_shard("train")
        test_dataset = train.get_dataset_shard("test")
        train_tf_dataset = to_tf_dataset(dataset=train_dataset, batch_size=config["batch_size"])
        test_tf_dataset = to_tf_dataset(dataset=test_dataset, batch_size=config["batch_size"])

        history = model.fit(train_tf_dataset, validation_data=test_tf_dataset, epochs=config["epochs"])
        results = history.history
    return results


def ray_train_tf(train_dataset, test_dataset):
    scaling_config = ScalingConfig(num_workers=4, use_gpu=False)
    train_config = {"use_gpu": False, "epochs": 3, "batch_size": 64, "num_workers": 4}

    trainer = TensorflowTrainer(
        train_loop_per_worker=train_loop_func,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        run_config=RunConfig(storage_path='/tmp/ray'),
        datasets={'train': train_dataset, 'test': test_dataset}
    )

    return trainer.fit()


if __name__ == "__main__":
    # In my project, the data is stored in delta spark, and I use DeltaTable and read_parquet to create ray.data.Dataset
    # For easy to duplicate the bug, I use the same data source and create ray.data.Dataset from local torchvision file (my
    # project support both pytorch and tensorflow framework).
    root_path = os.getcwd()
    data_path = os.path.join(root_path, "cifar-10-batches-py")
    if not os.path.exists(data_path):
        datasets.CIFAR10(root=root_path, train=True, download=True)
    train_data = torchvision_ray_dataset(data_path, is_train=True)
    test_data = torchvision_ray_dataset(data_path, is_train=False)
    result = ray_train_tf(train_data, test_data)
    print(result)

import os

import numpy as np
import tensorflow as tf
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
from ray import train
from ray.air import ScalingConfig, RunConfig, CheckpointConfig
from ray.air.integrations.keras import ReportCheckpointCallback
from ray.train.tensorflow import TensorflowTrainer, prepare_dataset_shard

from data_builder.image_data_builder import ImageDataBuilder


def build_and_train_ray_tf(model_class, train_data, test_data, config):
    scaling_config = ScalingConfig(num_workers=config['num_workers'], use_gpu=config['use_gpu'])
    datasets = {'train': train_data, 'test': test_data}
    config['model_class'] = model_class

    trainer = TensorflowTrainer(
        train_loop_per_worker=tf_train_worker_func,
        train_loop_config=config,
        scaling_config=scaling_config,
        run_config=RunConfig(checkpoint_config=CheckpointConfig(num_to_keep=2,
                                                                checkpoint_score_attribute="accuracy",
                                                                checkpoint_score_order="max")),
        datasets=datasets
    )

    result = trainer.fit()
    return result


def to_tf_dataset(dataset, batch_size):
    def to_tensor_iterator():
        # !!! FATAL, iter_tf_batches has critical bug, somtimes it will change the bytes array size, this cause
        # ImageDataBuilder.image_transform() transpose array failed.
        # for batch in dataset.iter_tf_batches(batch_size=batch_size):
        for batch in dataset.iter_batches(batch_size=batch_size):
            images = np.array(list(map(lambda x: ImageDataBuilder.image_transform(x), batch['image'])))
            images = images.transpose((0, 2, 3, 1))
            label = tf.cast(tf.expand_dims(batch["label"], 1), tf.int64)
            yield images, label

    output_signature = tf.TensorSpec(shape=(None, 32, 32, 3), dtype=tf.float32), tf.TensorSpec(shape=(None, 1), dtype=tf.int64)
    tf_dataset = tf.data.Dataset.from_generator(to_tensor_iterator, output_signature=output_signature)
    return prepare_dataset_shard(tf_dataset)


def tf_train_worker_func(config):
    # set TF_CONFIG to force use CPU, default use GPU has problem on Macbook M1 Pro
    # error: tensorflow/core/framework/op_kernel.cc:1803] INTERNAL: Failed to build OpKernel for Add : No registered
    # 'Add' OpKernel for 'GPU' devices compatible with node {{node Add}}
    # TODO: fix GPU problem to use GPU for distributed training.
    os.environ.pop('TF_CONFIG', None)

    batch_size = config.get("batch_size", 64)
    epochs = config.get("epochs", 3)

    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    results = {}
    with strategy.scope():
        model = config['model_class'](num_classes=10)
        train_data = train.get_dataset_shard("train")
        test_data = train.get_dataset_shard("test")
        train_data = to_tf_dataset(dataset=train_data, batch_size=batch_size)
        test_data = to_tf_dataset(dataset=test_data, batch_size=batch_size)
        # model_checkpoint_callback = ModelCheckpoint(
        #     filepath='./cifar10_ray_tensor_{epoch:02d}.keras',
        #     save_best_only=True,
        #     save_weights_only=True,
        #     monitor='val_accuracy',
        #     mode='max')
        # callbacks_list = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        #                   model_checkpoint_callback]
        callbacks_list = [ReportCheckpointCallback()]
        history = model.fit(train_data, validation_data=test_data, epochs=epochs, callbacks=callbacks_list)
        # history = model.fit(train_data, validation_data=test_data, epochs=epochs)
        results = history.history
    return results

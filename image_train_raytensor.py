import tensorflow as tf
from ray import train
from ray.air import ScalingConfig, RunConfig, CheckpointConfig
from ray.air.integrations.keras import ReportCheckpointCallback
from ray.train.tensorflow import TensorflowTrainer

from image_cnn_tensor import ImageTrainerTF, build_network


class ImageTrainerTFRay(ImageTrainerTF):
    def __init__(self, train_data, test_data, num_classes):
        super(ImageTrainerTFRay, self).__init__(train_data, test_data, num_classes)

    def save_checkpoint(self, epoch, train_acc, train_loss, test_acc):
        self.model.save("cifar10tf_{}.model".format(epoch))
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {}, Test Accuracy: {}".format(epoch, train_acc, train_loss, test_acc))

    @staticmethod
    def build_and_train(train_data, test_data, config):
        scaling_config = ScalingConfig(num_workers=config['num_workers'], use_gpu=config['use_gpu'])
        datasets = {'train': train_data, 'test': test_data}
        config['train_len'] = train_data.count()
        config['test_len'] = test_data.count()
        trainer = TensorflowTrainer(
            train_loop_per_worker=ImageTrainerTFRay.train_loop_per_worker,
            train_loop_config=config,
            scaling_config=scaling_config,
            run_config=RunConfig(checkpoint_config=CheckpointConfig(num_to_keep=2,
                                                                    checkpoint_score_attribute="test_acc",
                                                                    checkpoint_score_order="max"),
                                 storage_path=config['work_dir']),
            datasets=datasets
        )

        # Train the model.
        result = trainer.fit()
        return result

    @staticmethod
    def train_loop_per_worker(config):
        batch_size = config.get("batch_size", 64)
        epochs = config.get("epochs", 3)

        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        results = []
        with strategy.scope():
            net = build_network(num_classes=config['num_classes'])
            train_data = train.get_dataset_shard("train")
            test_data = train.get_dataset_shard("test")

            for _ in range(epochs):
                train_dataset = train_data.to_tf(feature_columns="image", label_columns="label", batch_size=batch_size)
                test_dataset = test_data.to_tf(feature_columns="image", label_columns="label", batch_size=batch_size)
                history = net.fit(train_dataset, validation_data=test_data, callbacks=[ReportCheckpointCallback()])
                results.append(history.history)
        return results

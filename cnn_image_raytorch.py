import numpy as np
from ray.air import RunConfig, CheckpointConfig, ScalingConfig
from ray.train import Checkpoint
from ray.train.torch import TorchTrainer
import ray.train.torch
from torch.nn.parallel import DistributedDataParallel
import torch

from cnn_image_torch import ImageTrainerTorch, ImageCnnTorch
from image_data_builder import ImageDataBuilder


class ImageTrainerTorchRay(ImageTrainerTorch):
    def __init__(self, model, batch_size=32):
        super(ImageTrainerTorchRay, self).__init__(model)
        self.batch_size = batch_size
        self.train_data = ray.train.get_dataset_shard('train')
        self.test_data = ray.train.get_dataset_shard('test')

    def prepare_model(self):
        self.model = ray.train.torch.prepare_model(self.model)

        self.train_data = self.train_data.iter_batches(batch_size=self.batch_size)
        self.test_data = self.test_data.iter_batches(batch_size=self.batch_size)

    def data_to_device(self, images, labels):
        return images, labels

    def save_checkpoint(self, epoch, train_acc, train_loss, test_acc):
        base_model = (self.model.module if isinstance(self.model, DistributedDataParallel) else self.model)
        torch.save(base_model.state_dict(), 'cifar10model_{}.model'.format(epoch))
        checkpoint = Checkpoint.from_directory('./')
        # Report metrics and checkpoint.
        ray.train.report({'train_acc': train_acc, 'train_loss': train_loss, 'test_acc': test_acc}, checkpoint=checkpoint)

    @staticmethod
    def build_and_train(image_data, config):
        scaling_config = ScalingConfig(num_workers=config['num_workers'], use_gpu=config['use_gpu'])
        batch_size = config['batch_size']
        num_workers = config['num_workers']
        train_data = image_data.load_to_ray_dateset(True)
        test_data = image_data.load_to_ray_dateset(False)
        datasets = {'train': train_data, 'test': test_data}
        trainer = TorchTrainer(
            train_loop_per_worker=ImageTrainerTorchRay.train_loop_per_worker,
            train_loop_config=config,
            scaling_config=scaling_config,
            run_config=RunConfig(checkpoint_config=CheckpointConfig(num_to_keep=1)),
            datasets=datasets
        )

        # Train the model.
        result = trainer.fit()
        return result

    @staticmethod
    def train_loop_per_worker(config):
        net = ImageCnnTorch(num_classes=config['num_classes'])
        model = ImageTrainerTorchRay(net, config['batch_size'])
        model.train(config['num_epochs'])

    @staticmethod
    def image_from_buffer(pic):
        image = ImageDataBuilder.image_transform(pic)
        return image

    def extract_item(self, item):
        images = np.array(list(map(ImageTrainerTorchRay.image_from_buffer, item['image'])))
        return torch.as_tensor(images), torch.as_tensor(item['label'])

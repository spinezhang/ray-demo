import os

import numpy as np
from ray.air import RunConfig, CheckpointConfig, ScalingConfig
from ray.train import Checkpoint
from ray.train.torch import TorchTrainer
import ray.train.torch
from torch.nn.parallel import DistributedDataParallel
import torch

from image_train_torch import ImageTrainerTorch, ImageTorchProcess
from data_builder.image_data_builder import ImageDataBuilder


class ImageTrainerTorchRay(ImageTrainerTorch):
    def __init__(self, model, config):
        super(ImageTrainerTorchRay, self).__init__(model, config['use_gpu'])
        self.batch_size = config['batch_size']
        self.train_data = ray.train.get_dataset_shard('train')
        self.test_data = ray.train.get_dataset_shard('test')
        self.train_len = config['train_len']
        self.test_len = config['test_len']
        self.work_dir = config['work_dir']

    def prepare_model(self):
        self.model = ray.train.torch.prepare_model(self.model)

        self.train_data = self.train_data.iter_batches(batch_size=self.batch_size, local_shuffle_buffer_size=50)
        self.test_data = self.test_data.iter_batches(batch_size=self.batch_size)

    def save_checkpoint(self, epoch, train_acc, train_loss, test_acc):
        base_model = (self.model.module if isinstance(self.model, DistributedDataParallel) else self.model)
        checkpoint = None
        # In standard DDP training, where the model is the same across all ranks,
        # only the global rank 0 worker needs to save and report the checkpoint
        if ray.train.get_context().get_world_rank() == 0:
            torch.save(
                base_model.state_dict(),  # NOTE: Unwrap the model.
                os.path.join(self.work_dir, 'cifar10_ray_torch.model'),
            )
            # checkpoint = Checkpoint.from_directory(self.work_dir)
        ray.train.report({'train_acc': train_acc, 'train_loss': train_loss, 'test_acc': test_acc}, checkpoint=checkpoint)

    @staticmethod
    def build_and_train(model, train_data, test_data, config):
        scaling_config = ScalingConfig(num_workers=config['num_workers'], use_gpu=config['use_gpu'])
        datasets = {'train': train_data, 'test': test_data}
        config['model'] = model
        config['train_len'] = train_data.count()
        config['test_len'] = test_data.count()
        trainer = TorchTrainer(
            train_loop_per_worker=ImageTrainerTorchRay.train_loop_per_worker,
            train_loop_config=config,
            scaling_config=scaling_config,
            run_config=RunConfig(checkpoint_config=CheckpointConfig(num_to_keep=2,
                                                                    checkpoint_score_attribute="test_acc",
                                                                    checkpoint_score_order="max"),
                                 storage_path=config['work_dir']),
            datasets=datasets
        )

        result = trainer.fit()
        return result

    @staticmethod
    def train_loop_per_worker(config):
        model = ImageTrainerTorchRay(config['model'], config)
        model.train(config['num_epochs'])

    def extract_item(self, item):
        images = np.array(list(map(ImageDataBuilder.image_transform, item['image'])))
        return torch.as_tensor(images), torch.as_tensor(item['label'])


class ImageTorchInferenceRay(ImageTorchProcess):
    def __init__(self, model, path, use_gpu=True):
        super().__init__(use_gpu)
        checkpoint = torch.load(path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})
        self.net = self.prepare_model(model)

    def prepare_model(self, model):
        model.to(self.device)
        model.eval()
        return model

    def __call__(self, batch_data):
        images = np.array(list(map(ImageDataBuilder.image_transform, batch_data['image'])))
        torch.inference_mode()
        outputs = self.net(torch.as_tensor(images))
        prediction = outputs.argmax(dim=1)
        correct_count = torch.sum(prediction == torch.as_tensor(batch_data['label']).data).item()
        return {'correct_count': np.array([correct_count])}

    def batch_predict(self, test_dataset, batch_size=200, num_workers=6):
        length = test_dataset.count()
        prediction = test_dataset.map_batches(self, batch_size=batch_size, compute=ray.data.ActorPoolStrategy(size=num_workers), zero_copy_batch=True)
        prediction.take_all()
        correct_count = prediction.sum('correct_count')
        return correct_count / length

import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from abc import abstractmethod


class ImageTorchProcess:
    def __init__(self, use_gpu=True):
        if not use_gpu:
            device = torch.device('cpu')
        else:
            cuda_avail = torch.cuda.is_available()
            if cuda_avail:
                device = torch.device('cuda')
            else:
                device = torch.device("mps")
                if device.type != 'mps':
                    device = torch.device('cpu')
        self.device = device
        self.use_gpu = use_gpu

    def extract_data_and_process(self, data_loader, process_func, *args):
        progress = tqdm(data_loader)
        for item in progress:
            images, labels = self.extract_item(item)
            images = Variable(images)
            images, labels = self.data_to_device(images, labels)
            args = process_func(images, labels, *args)
        return args

    def data_to_device(self, images, labels):
        return images.to(self.device), labels.to(self.device)

    def extract_item(self, item):
        return item[0], item[1]

    def test(self, model, test_loader, length):
        test_acc = 0.0
        _, test_acc = self.extract_data_and_process(test_loader, self.test_images, model, test_acc)
        # Compute the average acc and loss over all 10000 test images
        test_acc = test_acc / length
        return test_acc

    @staticmethod
    def test_images(images, labels, model, test_acc):
        outputs = model(images)
        prediction = outputs.argmax(dim=1)
        test_acc += torch.sum(prediction == labels.data).item()
        return model, test_acc


class ImageTrainerTorch(ImageTorchProcess):
    def __init__(self, model, use_gpu):
        super().__init__(use_gpu)
        self.model = model
        self.train_data = None
        self.test_data = None
        self.train_len = 0
        self.test_len = 0

    @abstractmethod
    def prepare_model(self):
        pass

    @abstractmethod
    def save_checkpoint(self, epoch, train_acc, train_loss, test_acc):
        pass

    def train(self, num_epochs):
        # Create model, optimizer and loss function
        self.prepare_model()
        optimizer = Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            self.model.train()
            train_acc = 0.0
            train_loss = 0.0

            _, optimizer, train_acc, train_loss = self.extract_data_and_process(self.train_data, self.train_images, loss_fn, optimizer, train_acc, train_loss)

            # Compute the average acc and loss over all 50000 training images
            train_acc = train_acc / self.train_len
            train_loss = train_loss / self.train_len

            # Evaluate on the test set
            self.model.eval()
            test_acc = self.test(self.model, self.test_data, self.test_len)

            # lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
            # scheduler.step()
            # Save the model if the test acc is greater than our current best
            self.save_checkpoint(epoch, train_acc, train_loss, test_acc)

    def train_images(self, images, labels, loss_fn, optimizer, train_acc, train_loss):
        # print(f"training image, label{labels}")
        optimizer.zero_grad()
        # Predict classes using images from the test set
        outputs = self.model(images)
        # Compute the loss based on the predictions and actual labels
        loss = loss_fn(outputs, labels)
        # Backpropagate the loss
        loss.backward()
        # Adjust parameters according to the computed gradients
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        _, prediction = torch.max(outputs.data, 1)
        train_acc += torch.sum(prediction == labels.data).item()
        return loss_fn, optimizer, train_acc, train_loss


class ImageTrainerTorchSingle(ImageTrainerTorch):
    def __init__(self, model, train_data, test_data, config):
        super(ImageTrainerTorchSingle, self).__init__(model, config['use_gpu'])
        self.train_data = train_data
        self.test_data = test_data
        self.train_len = len(train_data)
        self.test_len = len(test_data)
        self.batch_size = config['batch_size']
        self.num_workers = config["num_workers"]
        self.test_acc = 0

    def prepare_model(self):
        model = DataParallel(self.model)
        self.model = model.to(self.device)
        self.train_data = DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
        self.test_data = DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def save_checkpoint(self, epoch, train_acc, train_loss, test_acc):
        if test_acc > self.test_acc:
            torch.save(self.model.state_dict(), "cifar10_torch_single.model")
            self.test_acc = test_acc
        # Print the metrics
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {}, Test Accuracy: {}".format(epoch, train_acc, train_loss, test_acc))

    @staticmethod
    def build_and_train(model, train_data, test_data, config):
        trainer = ImageTrainerTorchSingle(model, train_data, test_data, config)
        trainer.train(config['num_epochs'])


class ImageTorchInferenceSingle(ImageTorchProcess):
    def __init__(self, model, path, use_gpu=True):
        super().__init__(use_gpu)
        checkpoint = torch.load(path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})
        self.model = model
        self.prepare_model()

    def prepare_model(self):
        model = DataParallel(self.model)
        self.model = model.to(self.device)

    def batch_predict(self, test_dataset):
        length = len(test_dataset)
        test_data = DataLoader(test_dataset, batch_size=50, num_workers=4)
        return self.test(self.model, test_data, length)

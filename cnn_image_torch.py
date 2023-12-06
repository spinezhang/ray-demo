import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from abc import abstractmethod


class ImageCnnTorch(nn.Module):
    def __init__(self, num_classes=10):
        super(ImageCnnTorch, self).__init__()

        # Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=3,out_channels=32)
        self.unit2 = Unit(in_channels=32, out_channels=32)
        self.unit3 = Unit(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=32, out_channels=64)
        self.unit5 = Unit(in_channels=64, out_channels=64)
        self.unit6 = Unit(in_channels=64, out_channels=64)
        self.unit7 = Unit(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit8 = Unit(in_channels=64, out_channels=128)
        self.unit9 = Unit(in_channels=128, out_channels=128)
        self.unit10 = Unit(in_channels=128, out_channels=128)
        self.unit11 = Unit(in_channels=128, out_channels=128)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit12 = Unit(in_channels=128, out_channels=128)
        self.unit13 = Unit(in_channels=128, out_channels=128)
        self.unit14 = Unit(in_channels=128, out_channels=128)

        self.avgpool = nn.AvgPool2d(kernel_size=4)

        # Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(self.unit1, self.unit2, self.unit3, self.pool1, self.unit4, self.unit5, self.unit6,
                                 self.unit7, self.pool2, self.unit8, self.unit9, self.unit10, self.unit11, self.pool3,
                                 self.unit12, self.unit13, self.unit14, self.avgpool)

        self.fc = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, 128)
        output = self.fc(output)
        return output


class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output


# Create a learning rate adjustment function that divides the learning rate by 10 every 30 epochs
def adjust_learning_rate(epoch, optimizer):
    lr = 0.001

    if epoch > 180:
        lr = lr / 1000000
    elif epoch > 150:
        lr = lr / 100000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 60:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class ImageTrainerTorch:
    def __init__(self, model):
        self.model = model
        self.train_data = None
        self.test_data = None

    @abstractmethod
    def prepare_model(self):
        pass

    @abstractmethod
    def data_to_device(self, images, labels):
        pass

    @abstractmethod
    def save_checkpoint(self, epoch, train_acc, train_loss, test_acc):
        pass

    def train(self, num_epochs):
        # Create model, optimizer and loss function
        self.prepare_model()
        optimizer = Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)

        # Define the optimizer and loss function
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            self.model.train()
            train_acc = 0.0
            train_loss = 0.0

            _, optimizer, train_acc, train_loss = self.extract_data_and_process(self.train_data, self.train_images, loss_fn, optimizer, train_acc, train_loss)

            # Call the learning rate adjustment function
            adjust_learning_rate(epoch, optimizer)

            # Compute the average acc and loss over all 50000 training images
            train_acc = train_acc / 50000
            train_loss = train_loss / 50000

            # Evaluate on the test set
            test_acc = self.test(self.test_data)

            # Save the model if the test acc is greater than our current best
            self.save_checkpoint(epoch, train_acc, train_loss, test_acc)

    def extract_data_and_process(self, data_loader, process_func, *args):
        progress = tqdm(data_loader)
        for item in progress:
            images, labels = self.extract_item(item)
            images = Variable(images)

            images, labels = self.data_to_device(images, labels)
            args = process_func(images, labels, *args)
        return args

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
        train_loss += loss.item()
        _, prediction = torch.max(outputs.data, 1)
        train_acc += torch.sum(prediction == labels.data).item()
        return loss_fn, optimizer, train_acc, train_loss

    def test(self, test_loader):
        self.model.eval()
        test_acc = 0.0
        test_acc, _ = self.extract_data_and_process(test_loader, self.test_images, test_acc, 0)

        # Compute the average acc and loss over all 10000 test images
        test_acc = test_acc / 10000

        return test_acc

    def test_images(self, images, labels, test_acc, dummy):
        outputs = self.model(images)
        _, prediction = torch.max(outputs.data, 1)
        test_acc += torch.sum(prediction == labels.data).item()
        return test_acc, dummy

    @abstractmethod
    def extract_item(self, item):
        pass


def get_device():
    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        return torch.device('cuda')
    device = torch.device("mps")
    if device.type != 'mps':
        return torch.device('cpu')
    return device


class ImageTrainerTorchSingle(ImageTrainerTorch):
    def __init__(self, model, train_data, test_data, use_gpu=False):
        super(ImageTrainerTorchSingle, self).__init__(model)
        self.train_data = train_data
        self.test_data = test_data
        if use_gpu:
            self.device = get_device()
        else:
            self.device = torch.device('cpu')

    def prepare_model(self):
        model = DataParallel(self.model)
        self.model = model.to(self.device)

    def data_to_device(self, images, labels):
        return images.to(self.device), labels.to(self.device)

    def save_checkpoint(self, epoch, train_acc, train_loss, test_acc):
        torch.save(self.model.state_dict(), "cifar10model_{}.model".format(epoch))
        # Print the metrics
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {}, Test Accuracy: {}".format(epoch, train_acc, train_loss, test_acc))

    @staticmethod
    def build_and_train(image_data, config):
        batch_size = config['batch_size']
        num_workers = config['num_workers']
        train_data = DataLoader(image_data.load_to_torch_dateset(is_train=True), batch_size=batch_size, num_workers=num_workers)
        test_data = DataLoader(image_data.load_to_torch_dateset(is_train=False), batch_size=batch_size, num_workers=num_workers)

        net = ImageCnnTorch(num_classes=config['num_classes'])
        model = ImageTrainerTorchSingle(net, train_data, test_data, config['use_gpu'])
        model.train(config['num_epochs'])

    def extract_item(self, item):
        return item[0], item[1]

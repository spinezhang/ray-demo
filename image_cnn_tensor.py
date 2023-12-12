from abc import abstractmethod
import numpy as np
from keras import layers, models, metrics
import tensorflow as tf
from tensorflow.python.framework.ops import SymbolicTensor


def build_network(num_classes=10):
    model = models.Sequential([
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
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def _fixup_shape(image, label):
    label = tf.expand_dims(label, 1)
    label.set_shape((None, 1))
    image.set_shape((None, 3, 32, 32))
    image = tf.transpose(image, (0, 2, 3, 1))
    return image, label


class ImageTrainerTF:
    def __init__(self, train_data, test_data, num_classes=10):
        tf.data.experimental.enable_debug_mode()
        self.model = build_network(num_classes)
        self.train_data = train_data
        self.test_data = test_data

    def train(self, batch_size=32, epochs=50):
        self.train_data = self.train_data.batch(batch_size).map(_fixup_shape)
        self.test_data = self.test_data.batch(batch_size).map(_fixup_shape)
        self.model.fit(self.train_data, validation_data=self.test_data, epochs=epochs, shuffle=True)

    def test(self):
        return self.model.evaluate(self.test_data)

    def inference(self, image):
        predictions = self.model.predict(self.test_data.map(lambda images, labels: images))
        predictions = np.argmax(predictions, axis=1)
        return predictions

    def save(self, file):
        self.model.save(file)

    @abstractmethod
    def save_checkpoint(self, epoch, train_acc, train_loss, test_acc):
        pass


class ImageTrainerTFSingle(ImageTrainerTF):
    def __init__(self, train_data, test_data, num_classes):
        super(ImageTrainerTFSingle, self).__init__(train_data, test_data, num_classes)

    def save_checkpoint(self, epoch, train_acc, train_loss, test_acc):
        self.model.save("cifar10tf_{}.model".format(epoch))
        # Print the metrics
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {}, Test Accuracy: {}".format(epoch, train_acc, train_loss, test_acc))

    @staticmethod
    def build_and_train(train_data, test_data, config):
        batch_size = config['batch_size']
        num_workers = config['num_workers']

        model = ImageTrainerTFSingle(train_data, test_data, config['num_classes'])
        model.train(batch_size, config['num_epochs'])
        score = model.test()
        print(f"score={score}")

    @staticmethod
    def tensor_image_transform(row):
        if isinstance(row.image, SymbolicTensor):
            image = tf.reshape(tf.convert_to_tensor(np.zeros(3072, dtype=np.float32)), shape=(-1, 32, 32, 3), name='args')
            label = tf.convert_to_tensor(np.zeros(1, dtype=np.int64))
        else:
            image = row.image.numpy()
            image = np.array([np.frombuffer(image, dtype=np.uint8)]).astype(np.float32) / 255.0
            image = image.reshape(-1, 32, 32, 3)
            image = tf.convert_to_tensor(image)
            label = image.label.reshape(1)
        return image, label


import numpy as np
import tensorflow as tf


class ImageTrainerTF:
    def __init__(self, model, train_data, test_data):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data

    def train(self, epochs=50):
        self.model.fit(self.train_data, validation_data=self.test_data, epochs=epochs, shuffle=True)

    def test(self):
        return self.model.evaluate(self.test_data)

    def inference(self, image):
        predictions = self.model.predict(self.test_data.map(lambda images, labels: images))
        predictions = np.argmax(predictions, axis=1)
        return predictions

    def save(self, file):
        self.model.save(file)


def fix_dataset_shape(image, label):
    label = tf.expand_dims(label, 1)
    label.set_shape((None, 1))
    image.set_shape((None, 3, 32, 32))
    image = tf.transpose(image, (0, 2, 3, 1))
    return image, label


class ImageTrainerTFSingle(ImageTrainerTF):
    def __init__(self, model, train_data, test_data):
        super(ImageTrainerTFSingle, self).__init__(model, train_data, test_data)

    def save_checkpoint(self, epoch, train_acc, train_loss, test_acc):
        self.model.save("cifar10tf_{}.model".format(epoch))
        # Print the metrics
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {}, Test Accuracy: {}".format(epoch, train_acc, train_loss, test_acc))

    @staticmethod
    def build_and_train(model, train_data, test_data, config):
        batch_size = config['batch_size']

        train_data = train_data.batch(batch_size).map(fix_dataset_shape)
        test_data = test_data.batch(batch_size).map(fix_dataset_shape)
        trainer = ImageTrainerTFSingle(model, train_data, test_data)
        trainer.train(config['num_epochs'])
        accuracy = trainer.test()
        print(f"accuracy:{accuracy}")

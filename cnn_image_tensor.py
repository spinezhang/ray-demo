from keras import layers, models


class ImageCnnTensor:
    def __init__(self, num_classes=10):
        self.model = models.Sequential([
            layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
            layers.BatchNormalization(),

            layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            # layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
            # layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            # layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
            # layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            # layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
            # layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dropout(0.2),

            # Hidden layer
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.2),

            layers.Dense(10, activation='softmax')
        ])


model = Model(i, x)

# model description
model.summary()

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

r = model.fit(
  x_train, y_train, validation_data=(x_test, y_test), epochs=50)

batch_size = 32
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

train_generator = data_generator.flow(x_train, y_train, batch_size)
steps_per_epoch = x_train.shape[0] // batch_size

r = model.fit(train_generator, validation_data=(x_test, y_test),
              steps_per_epoch=steps_per_epoch, epochs=50)


# label mapping

labels = '''airplane automobile bird cat deerdog frog horseship truck'''.split()

# select the image from our test dataset
image_number = 0

# display the image
plt.imshow(x_test[image_number])

# load the image in an array
n = np.array(x_test[image_number])

# reshape it
p = n.reshape(1, 32, 32, 3)

# pass in the network for prediction and
# save the predicted label
predicted_label = labels[model.predict(p).argmax()]

# load the original label
original_label = labels[y_test[image_number]]

# display the result
print("Original label is {} and predicted label is {}".format(
	original_label, predicted_label))

model.save('geeksforgeeks.h5')


---------------
# NHWC: (1, 725, 1920, 3)
predict_image = tf.expand_dims(image, 0)
# NCHW: (1, 3, 725, 1920)
image = np.transpose(tf.expand_dims(image, 0).numpy(), [0, 3, 1, 2])

# get transferred torch ResNet18 with pre-trained ImageNet weights
model = converted_fully_convolutional_resnet18(
    input_tensor=image, pretrained_resnet=True,
)

# Perform inference.
# Instead of a 1×1000 vector, we will get a
# 1×1000×n×m output ( i.e. a probability map
# of size n × m for each 1000 class,
# where n and m depend on the size of the image).
preds = model.predict(predict_image)
# NHWC: (1, 3, 8, 1000) back to NCHW: (1, 1000, 3, 8)
preds = tf.transpose(preds, (0, 3, 1, 2))
preds = tf.nn.softmax(preds, axis=1)

--------------
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value
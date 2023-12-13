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

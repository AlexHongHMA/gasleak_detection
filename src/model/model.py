from keras import layers, models

def cnn_3d_model(input_shape, num_classes=2):
    model = models.Sequential()

    # Input Layer
    model.add(layers.Input(shape=input_shape))

    # Conv-Pool Structure 1
    model.add(layers.Conv3D(4, (5, 5, 5), activation='relu', padding='same'))
    model.add(layers.LayerNormalization())
    model.add(layers.MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(layers.Dropout(0.1))

    # Conv-Pool Structure 2
    model.add(layers.Conv3D(8, (5, 5, 5), activation='relu', padding='same'))
    model.add(layers.LayerNormalization())
    model.add(layers.MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(layers.Dropout(0.1))

    # Conv-Pool Structure 3
    model.add(layers.Conv3D(16, (5, 5, 5), activation='relu', padding='same'))
    model.add(layers.LayerNormalization())
    model.add(layers.MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(layers.Dropout(0.1))

    # Conv-Pool Structure 4
    model.add(layers.Conv3D(32, (5, 5, 5), activation='relu', padding='same'))
    model.add(layers.LayerNormalization())
    model.add(layers.MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(layers.Dropout(0.1))

    # Flatten the output before feeding into fully connected layers
    model.add(layers.Flatten())

    # Fully Connected Layer 1 with Dropout
    model.add(layers.Dense(640, activation='relu'))
    model.add(layers.Dropout(0.3))  

    # Fully Connected Layer 2 with Dropout
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))  

    # Output Layer with 2 classes
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()  # Print model summary
    return model
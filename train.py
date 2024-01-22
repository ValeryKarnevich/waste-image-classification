import numpy as np
import scipy
import tensorflow as tf
import tensorflow.keras as keras
# from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50


def train_model():
    # Image data directory
    data_dir_path = 'data/'

    # Training data parameters
    batch_size = 32
    img_height = 299
    img_width = 299

    # Train and validation datasets
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input_resnet50,
        validation_split=0.2,
        height_shift_range=0.2,
    )

    train_ds = train_datagen.flow_from_directory(
        data_dir_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        seed=23,
        subset='training')

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input_resnet50,
        validation_split=0.2)

    val_ds = val_datagen.flow_from_directory(
        data_dir_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=23,
        subset='validation')

    # ResNet50 base model with fixed weights
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(img_width, img_height, 3)
    )

    base_model.trainable = False

    # Hyperparameter values
    learning_rate = 0.0005
    size_inner = 100
    droprate = 0.0

    # Neural network structure
    inputs = keras.Input(shape=(img_width, img_height, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner)
    outputs = keras.layers.Dense(6, activation='softmax')(drop)
    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=False)

    # Model compilation
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    print(f'''
    Compiled ResNet50 model with 
    learning rate = {learning_rate},
    size of additional inner layer = {size_inner},
    dropout rate = {droprate}
    ''')

    # Callback for saving the best model
    model_filename = 'resnet50_best_model.h5'
    checkpoint = keras.callbacks.ModelCheckpoint(
        model_filename,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )

    # Model training (saving best one within 10 epochs)
    epochs = 10
    history = model.fit(train_ds, epochs=epochs, validation_data=val_ds,
                       callbacks=[checkpoint])

    # Extracting best accuracy
    best_accuracy = max(history.history['val_accuracy'])

    print(f'''Training finished.
    Best model with validation accuracy of {best_accuracy} 
    saved to file {model_filename}''')


if __name__ == '__main__':
    train_model()


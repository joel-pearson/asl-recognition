from __future__ import absolute_import, division, print_function, unicode_literals

# imports
import os
import pathlib
import tensorflow as tf
import matplotlib.pylab as plt
import tensorflow_hub as hub
import numpy as np
import splitfolders
from PIL import Image as Image


image_shape = (224, 224)  # assigns a size to the variable IMAGE_SHAPE


# Split data into training and testing data
def train_test_split():
    splitfolders.ratio(input='./data/dataset/unprocessed_data', output='./data/dataset/processed_data', seed=1337,
                        ratio=(.8, .2))


# Processes the data for model to be trained on
def data_pre_processing():
    global image_batch_size, training_dataset, testing_dataset
    train_data_dir = './data/dataset/processed_data/train'
    train_data_dir = pathlib.Path(train_data_dir)
    test_data_dir = './data/dataset/processed_data/val'
    test_data_dir = pathlib.Path(test_data_dir)
    class_names = np.array([item.name for item in train_data_dir.glob('*') if item.name != 'LICENSE.txt'])

    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)

    training_dataset = train_generator.flow_from_directory(str(train_data_dir), target_size=image_shape, shuffle=True,
                                                           classes=list(class_names))
    testing_dataset = test_generator.flow_from_directory(str(test_data_dir), target_size=image_shape, shuffle=True)

    with open('./data/labels.txt', 'w') as f:
        for item in class_names:
            f.write("%s\n" % item)

    for image_batch_size, label_batch in training_dataset:
        print("Image batch shape: ", image_batch_size.shape)
        print("Label batch shape: ", label_batch.shape)
        break

    for image_batch_size, label_batch in testing_dataset:
        print("Image batch shape: ", image_batch_size.shape)
        print("Label batch shape: ", label_batch.shape)
        break


def create_checkpoint_callback():
    global checkpoint_dir, cp_callback
    checkpoint_path = './model/checkpoints/cp.ckpt'  # save checkpoint to directory
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)


def create_early_stop():
    global es_callback
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        mode='min',
        verbose=1,
        patience=5,
        restore_best_weights=True,
        min_delta=0.001,
    )


# Creates and trains the model
def create_model():
    feature_vector_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4"
    feature_vector_layer = hub.KerasLayer(feature_vector_url, input_shape=(224, 224, 3))
    feature_vector_layer.trainable = False

    model = tf.keras.Sequential([
        feature_vector_layer,
        tf.keras.layers.experimental.preprocessing.Normalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(training_dataset.num_classes, activation='softmax')
    ])

    model.summary()

    model.compile(
        optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    epochs = 5
    steps_per_epoch = np.ceil(training_dataset.samples / training_dataset.batch_size)
    testing_steps = np.ceil(testing_dataset.samples / testing_dataset.batch_size)

    history = model.fit(
        training_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=testing_dataset,
        validation_steps=testing_steps,
        callbacks=[cp_callback]
    )

    model.save('./model/model-bs.h5')

    if not os.path.exists("./data/plots"):
        os.makedirs("./data/plots")

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.savefig('./data/plots/accuracy_epochs_bs.png')

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('./data/plots/loss_epochs_bs.png')


if os.path.exists('./data/dataset/processed_data'):
    pass
else:
    train_test_split()

data_pre_processing()
create_checkpoint_callback()
create_model()

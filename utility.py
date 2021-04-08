import numpy as np
import keras
from tensorflow.keras import layers
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

def store_model(model, name):
    model_json = model.to_json()
    with open('data/' + name + '.json', 'w') as data_file:
        data_file.write(model_json)
    model.save_weights('data/' + name + '.h5')

def load_model(name):
    json_file = open('data/' + name + '.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights('data/' + name + '.h5')
    return model

def get_dataset(dataset):
    if dataset == "mnist":
        image_size = 28
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        y_test = to_categorical(y_test, 10)
        y_train = to_categorical(y_train, 10)
    elif dataset == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        y_test = to_categorical(y_test, 10)
        y_train = to_categorical(y_train, 10)
    elif dataset == 'cifer10':
        image_size = 32
        latent_units_size = 20
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        y_test = to_categorical(y_test, 10)
        y_train = to_categorical(y_train, 10)
    elif dataset == 'pets':
        image_size = (90, 90)
        num_images = 10000

        ds = tf.keras.preprocessing.image_dataset_from_directory(
                "data/pets/", validation_split=0, seed=1337, image_size=image_size, batch_size=num_images, shuffle=True)

        normalize = layers.experimental.preprocessing.Rescaling(1./ 255)
        normalized_ds = ds.map(lambda x, y: (normalize(x), y))
        images, labels = next(iter(normalized_ds))

        validation_split = 0.8
        x_train = np.empty((int(len(images) * validation_split), 90, 90, 3))
        y_train = np.empty(int(len(images) * validation_split))
        train_len = int(len(images) * validation_split) + 1
        val_len = int(len(images) * (1.0 - validation_split))
        x_test = np.empty((val_len, 90, 90, 3), dtype='uint8')
        y_test = np.empty((val_len))

        for i in range(len(images)):
            if labels[i].numpy().astype(int).item() == 0:
                if i < int(len(images) * validation_split):
                    x_train[i] = images[i].numpy()
                    #plt.imshow(x_train[i])
                    #plt.show()
                    #quit()
                    y_train[i] = 0
                else:
                    x_test[i - train_len] = images[i].numpy()
                    y_test[i - train_len] = 0
            elif labels[i].numpy().astype(int).item() == 1:
                if i < int(len(images) * validation_split):
                    x_train[i] = images[i].numpy()
                    y_train[i] = 1
                else:
                    x_test[i - train_len] = images[i].numpy()
                    y_test[i - train_len] = 1

        y_train = to_categorical(y_train, 2)
        y_test = to_categorical(y_test, 2)

    return (x_train, y_train), (x_test, y_test)

def split_dataset(x_train, y_train):
    # Unlabled
    x_train_D1 = x_train[0 : len(x_train) - len(x_train) // 5] # 2 Funker med mnist. 30 Funker med cifer
    y_train_D1 = None
    # Labled
    x_train_D2 = x_train[len(x_train) - len(x_train) // 5:]
    y_train_D2 = y_train[len(y_train) - len(y_train) // 5 :]
    return (x_train_D1, y_train_D1), (x_train_D2, y_train_D2)

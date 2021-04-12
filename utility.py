import numpy as np
import keras
from tensorflow.keras import layers
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import sklearn

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
        x_data = np.concatenate((x_test, x_train))
        y_test = to_categorical(y_test, 10)
        y_train = to_categorical(y_train, 10)
        y_data = np.concatenate((y_test, y_train))
        x_data, y_data = sklearn.utils.shuffle(x_data, y_data, random_state=0)
        return x_data, y_data
    elif dataset == "fashion_mnist":
        image_size = 28
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        x_data = np.concatenate((x_test, x_train))
        y_test = to_categorical(y_test, 10)
        y_train = to_categorical(y_train, 10)
        y_data = np.concatenate((y_test, y_train))
        return x_data, y_data
    elif dataset == 'cifar10':
        image_size = 32
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_data = np.concatenate((x_test, x_train))
        y_test = to_categorical(y_test, 10)
        y_train = to_categorical(y_train, 10)
        y_data = np.concatenate((y_test, y_train))
        return x_data, y_data
    elif dataset == 'digits':
        data = sklearn.datasets.load_digits()
        x_data = data.data
        y_data = data.target
        x_data = x_data.astype('float32') / 16.0
        y_data = to_categorical(y_data, 10)
        return x_data, y_data

def split_dataset(x_data, y_data, D1D2_ratio, D2_training_ratio):
    D1 = x_data[0 : len(x_data) - len(x_data) // D1D2_ratio]
    x_D2 = x_data[len(x_data) - len(x_data) // D1D2_ratio : ]
    y_D2 = y_data[len(y_data) - len(y_data) // D1D2_ratio : ]
    x_train_D2 = x_D2[0 : len(x_D2) - len(x_D2) // D2_training_ratio]
    y_train_D2 = y_D2[0 : len(y_D2) - len(y_D2) // D2_training_ratio]
    x_test_D2 = x_D2[len(x_D2) - len(x_D2) // D2_training_ratio : ]
    y_test_D2 = y_D2[len(y_D2) - len(y_D2) // D2_training_ratio : ]
    print("D1 length: " + str(len(D1)))
    print("D2 length: " + str(len(x_D2)))
    print("D2 training length: " + str(len(x_train_D2)))
    print("D2 validation length: " + str(len(x_test_D2)))
    return D1, (x_train_D2, y_train_D2), (x_test_D2, y_test_D2)

    if dataset == 'mnist':
        ratio = 10
    elif dataset == 'fashion_mnist':
        ratio = 10
    elif dataset == 'cifar10':
        ratio = 30
    elif dataset == 'digits':
        ratio = 10

    # Unlabled
    x_train_D1 = x_train[0 : len(x_train) - len(x_train) // ratio]
    y_train_D1 = None

    # Labled
    x_train_D2 = x_train[len(x_train) - len(x_train) // ratio:]
    y_train_D2 = y_train[len(y_train) - len(y_train) // ratio:]
    return (x_train_D1, y_train_D1), (x_train_D2, y_train_D2)

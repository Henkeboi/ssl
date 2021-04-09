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
    elif dataset == 'digits':
        data = sklearn.datasets.load_digits()
        x = data.data
        y = data.target
        x_train = x[0:1000]
        y_train = y[0:1000]
        x_test = x[1000:]
        y_test = y[1000:]
        x_train = x_train.astype('float32') / 16.0
        x_test = x_test.astype('float32') / 16.0
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

def split_dataset(x_train, y_train, dataset):
    if dataset == 'mnist':
        ratio = 10
    elif dataset == 'fashion_mnist':
        ratio = 10
    elif dataset == 'cifer10':
        ratio = 10
    elif dataset == 'digits':
        ratio = 10
    # Unlabled
    x_train_D1 = x_train[0 : len(x_train) - len(x_train) // ratio]
    y_train_D1 = None
    # Labled
    x_train_D2 = x_train[len(x_train) - len(x_train) // ratio:]
    y_train_D2 = y_train[len(y_train) - len(y_train) // ratio:]
    return (x_train_D1, y_train_D1), (x_train_D2, y_train_D2)

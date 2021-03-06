import cv2
import tensorflow as tf
import data
import utility
import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, Conv2DTranspose, Flatten, MaxPooling2D, Dropout, Reshape, BatchNormalization, UpSampling2D
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.optimizers import SGD
import utility
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn
import seaborn as sns


from sklearn.datasets import load_iris

class Encoder:
    def __init__(self, dataset, latent_size):
        self.dataset = dataset
        if dataset == 'mnist':
            image_size = 28
            input_shape = (image_size * image_size)
            layers = Input(shape=input_shape)
            self.input_layer = layers
            layers = Dense(784, activation='relu')(layers)
            layers = Dense(120, activation='relu')(layers)
            self.latent_layer = Dense(latent_size, activation='relu')(layers) # Latent layer
        elif dataset == 'fashion_mnist':
            image_size = 28
            input_shape = (image_size * image_size)
            layers = Input(shape=input_shape)
            self.input_layer = layers
            layers = Dense(784, activation='relu')(layers)
            layers = Dense(120, activation='relu')(layers)
            self.latent_layer = Dense(latent_size, activation='relu')(layers) # Latent layer
        elif dataset == 'cifar10':
            input_shape = Input(shape=(32, 32, 3))
            layers = input_shape
            self.input_layer = layers
            layers = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(layers)
            layers = BatchNormalization()(layers)
            layers = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(layers)
            layers = BatchNormalization()(layers)
            self.latent_layer = Dense(latent_size, activation='relu')(layers)
        elif dataset == 'digits':
            image_size = 8
            input_shape = (image_size * image_size)
            layers = Input(shape=input_shape)
            self.input_layer = layers
            layers = Dense(image_size * image_size, activation='relu')(layers)
            layers = Dense(60, activation='relu')(layers)
            self.latent_layer = Dense(latent_size, activation='relu')(layers) # Latent layer
        self.encoder = Model(self.input_layer, self.latent_layer)
        self.encoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def freeze(self):
        for layer in self.encoder.layers:
            layer.trainable = False
        self.encoder.layers[-1].trainable = True # The latent layer is trainable

    def get_latent_layer(self):
        return self.latent_layer

    def compile(self):
        self.encoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def get_input_layer(self):
        return self.input_layer
    
    def get_encoder(self):
        return self.encoder
    
    def plot_tSNE(self, num, title):
        self.encoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        if self.dataset == 'mnist':
            (x_train, y_train), (_ , _) = keras.datasets.mnist.load_data()
            x_train = x_train[:num].reshape(num, 28 ** 2)
            x_train = self.encoder.predict(x_train)
            y_train = y_train[:num]

            tsne = TSNE(n_components=2, verbose=1, random_state=123)
            z = tsne.fit_transform(x_train)
            df = pd.DataFrame()
            df["y"] = y_train
            df["1"] = z[:,0]
            df["2"] = z[:,1]

            plt.figure()
            sns.scatterplot(x="1", y="2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 10),
                data=df).set(title=title + " tSNE Mnist")
        elif self.dataset == 'fashion_mnist':
            (x_train, y_train), (_ , _) = keras.datasets.fashion_mnist.load_data()
            x_train = x_train[:num].reshape(num, 28 ** 2)
            x_train = self.encoder.predict(x_train)
            y_train = y_train[:num]

            tsne = TSNE(n_components=2, verbose=1, random_state=123)
            z = tsne.fit_transform(x_train)
            df = pd.DataFrame()
            df["y"] = y_train
            df["1"] = z[:,0]
            df["2"] = z[:,1]

            plt.figure()
            sns.scatterplot(x="1", y="2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 10),
                data=df).set(title=title + " tSNE Fashion Mnist")
        elif self.dataset == 'cifar10':
            (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data() 
            x_train = x_train[:num].reshape(num, 32, 32, 3)
            x_train = self.encoder.predict(x_train)
            y_train = y_train[:num]
            x_mnist = np.reshape(x_train, [x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3]])

            tsne = TSNE(n_components=2, verbose=1, random_state=123)
            z = tsne.fit_transform(x_mnist)
            df = pd.DataFrame()
            df["y"] = y_train.reshape(num)
            df["comp-1"] = z[:,0]
            df["comp-2"] = z[:,1]

            x_mnist = np.reshape(x_mnist, [x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3]])
            plt.figure()
            sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 10),
                data=df).set(title=title + " tSNE Cifar10")
        elif self.dataset == 'digits':
            data = sklearn.datasets.load_digits()
            x = data.data
            y = data.target
            x = x[0:num]
            y = y[0:num]
            x = self.encoder.predict(x)
            x = np.reshape(x, [x.shape[0], x.shape[1]])

            tsne = TSNE(n_components=2, verbose=1, random_state=123)
            z = tsne.fit_transform(x)
            df = pd.DataFrame()
            df["y"] = y
            df["comp-1"] = z[:,0]
            df["comp-2"] = z[:,1]

            x_mnist = np.reshape(x, [x.shape[0], x.shape[1]])
            plt.figure()
            sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 10),
                data=df).set(title=title + " tSNE Digits")


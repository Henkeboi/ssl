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

class Encoder:
    def __init__(self, dataset):
        self.dataset = dataset
        if dataset == 'mnist':
            image_size = 28
            input_shape = (image_size * image_size)
            layers = Input(shape=input_shape)
            self.input_layer = layers
            layers = Dense(784, activation='relu')(layers)
            layers = Dense(120, activation='relu')(layers)
            self.latent_layer = Dense(1000, activation='relu')(layers) # Latent layer
            self.encoder = Model(self.input_layer, self.latent_layer)
            self.encoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        elif dataset == 'fashion_mnist':
            image_size = 28
            input_shape = (image_size * image_size)
            layers = Input(shape=input_shape)
            self.input_layer = layers
            layers = Dense(784, activation='relu')(layers)
            layers = Dense(120, activation='relu')(layers)
            self.latent_layer = Dense(1000, activation='relu')(layers) # Latent layer
        elif dataset == 'cifer10':
            input_shape = Input(shape=(32, 32, 3))
            layers = input_shape
            self.input_layer = layers
            layers = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(layers)
            layers = BatchNormalization()(layers)
            layers = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(layers)
            layers = BatchNormalization()(layers)
            self.latent_layer = Dense(100, activation='relu')(layers)
            self.encoder = Model(self.input_layer, self.latent_layer)
            self.encoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        elif dataset == 'pets':
            input_shape = Input(shape=(90, 90, 3))
            layers = input_shape
            self.input_layer = layers
            layers = Conv2D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(layers)
            layers = BatchNormalization()(layers)
            layers = Conv2D(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(layers)
            layers = BatchNormalization()(layers)
            self.latent_layer = Dense(100, activation='relu')(layers)
            self.encoder = Model(self.input_layer, self.latent_layer)
            self.encoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    
    def is_trainable(trainable):
        for layer in self.encoder.layers:
            layer.trainable = trainable
        self.encoder.layers[-1].trainable = True # The latent layer is trainable

    def get_latent_layer(self):
        return self.latent_layer

    def compile(self):
        self.encoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def get_input_layer(self):
        return self.input_layer
    
    def get_encoder(self):
        return self.encoder
    
    def show_tSNE(self, input_images):

        latent_vectors = np.ndarray([])
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        for i in range(0, 10):
            input_image1 = input_images[i].reshape(1, 28 * 28)
            latent_vectors = np.append(self.encoder.predict(input_image1), latent_vectors)

        df = pd.DataFrame(latent_vectors)
        X_2d = tsne.fit_transform(df)
        for i in range(0, 10):
            plt.scatter(i, X_2d[i])
        plt.show()

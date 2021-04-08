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
from encoder import Encoder
from classifier import Classifier

class Autoencoder:
    def __init__(self, encoder, do_training, store_parameters_after_training, model_name):
        self.do_training = do_training
        self.store_parameters_after_training = store_parameters_after_training
        self.model_name = model_name
        self.encoder = encoder
        if encoder.dataset == "mnist":
            self.epochs = 2
            self.batch_size = 10
            latent_layer = encoder.get_latent_layer()
            layers = Dense(120, activation='relu')(latent_layer)
            layers = Dense(784, activation='sigmoid')(layers)
            self.autoencoder = Model(encoder.get_input_layer(), layers)
            self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        elif encoder.dataset == 'fashion_mnist':
            self.epochs = 1
            self.batch_size = 100
            latent_layer = encoder.get_latent_layer()
            layers = Dense(120, activation='relu')(latent_layer)
            layers = Dense(784, activation='sigmoid')(layers)
            self.autoencoder = Model(encoder.get_input_layer(), layers)
            self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        elif encoder.dataset == 'cifer10':
            self.epochs = 1
            self.batch_size = 100
            self.image_size = 32
            latent_layer = encoder.get_latent_layer()
            layers = UpSampling2D()(latent_layer)
            layers = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(layers)
            layers = BatchNormalization()(layers)
            layers = Conv2D(3,  kernel_size=1, strides=1, padding='same', activation='sigmoid')(layers)
            self.autoencoder = Model(encoder.get_input_layer(), layers)
            self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        elif encoder.dataset == 'pets':
            self.epochs = 1
            self.batch_size = 100
            self.image_size = 90
            latent_layer = encoder.get_latent_layer()
            layers = UpSampling2D()(latent_layer)
            layers = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(layers)
            layers = BatchNormalization()(layers)
            layers = Conv2D(3,  kernel_size=1, strides=1, padding='same', activation='sigmoid')(layers)
            self.autoencoder = Model(encoder.get_input_layer(), layers)
            self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def get_autoencoder(self):
        return self.encoder

    def train(self, x_train):
        if self.do_training == True:
            self.autoencoder.fit(x_train, x_train, epochs=self.epochs, batch_size=self.batch_size)
            self.encoder.compile()
            if self.store_parameters_after_training == True:
                utility.store_model(self.autoencoder, self.model_name)
        else:
            self.autoencoder = utility.load_model(self.model_name)
            self.encoder.compile()
            self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def show_reconstruction(self, input_image):
        if self.encoder.dataset == 'mnist' or self.encoder.dataset == 'fashion_mnist':  
            input_image = input_image[0].reshape(1, 28 * 28)
            output_image = self.autoencoder.predict(input_image)
            input_image = input_image[0]
            input_image = input_image.reshape(28, 28, 1)
            output_image = output_image.reshape(28, 28, 1)
        elif self.encoder.dataset == 'cifer10':
            input_image = input_image[0].reshape(1, 32, 32, 3)
            output_image = self.autoencoder.predict(input_image)
            input_image = input_image[0]
            input_image = input_image.reshape(32, 32, 3)
            output_image = output_image.reshape(32, 32, 3)
        elif self.encoder.dataset == 'pets':
            input_image = input_image[0].reshape(1, 180, 180, 3)
            output_image = self.autoencoder.predict(input_image)
            input_image = input_image[0]
            input_image = input_image.reshape(180, 180, 3)
            output_image = output_image.reshape(180, 180, 3)

        plt.figure(figsize=(5, 5))
        ax = plt.subplot(2, 1, 1)
        plt.axis('off')
        ax.set_title('Input image')
        plt.imshow(input_image)
        ax = plt.subplot(2, 1, 2)
        plt.axis('off')
        ax.set_title('Output image')
        plt.imshow(output_image)
        plt.show()

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

class Classifier:
    def __init__(self, encoder, la, loss_function, optimizer, epochs, do_training, store_parameters_after_training, model_name):
        self.do_training = do_training
        self.store_parameters_after_training = store_parameters_after_training
        self.model_name = model_name
        self.encoder = encoder
        if encoder.dataset == 'mnist':
            self.epochs = epochs
            self.batch_size = 100
            input_layer = encoder.get_input_layer()
            latent_layer = encoder.get_latent_layer()
            classifier_layer = Dense(10, activation='sigmoid')(latent_layer) 
        elif encoder.dataset == 'fashion_mnist':
            self.epochs = epochs
            self.batch_size = 100
            input_layer = encoder.get_input_layer()
            latent_layer = encoder.get_latent_layer()
            classifier_layer = Dense(10, activation='sigmoid')(latent_layer) 
        elif encoder.dataset == 'cifar10':
            self.epochs = epochs
            self.batch_size = 10
            latent_layer = encoder.get_latent_layer()
            classifier_layer = Flatten()(latent_layer) 
            classifier_layer = Dense(10, activation='sigmoid')(classifier_layer) 
            input_layer = encoder.get_input_layer()
        elif encoder.dataset == 'digits':
            self.epochs = epochs
            self.batch_size = 1
            latent_layer = encoder.get_latent_layer()
            classifier_layer = Flatten()(latent_layer) 
            classifier_layer = Dense(10, activation='sigmoid')(classifier_layer) 
            input_layer = encoder.get_input_layer()
        self.classifier = Model(input_layer, classifier_layer)
        if optimizer == 'Adam':
            opt = keras.optimizers.Adam(learning_rate=la)
        if optimizer == 'SGD':
            opt = keras.optimizers.SGD(learning_rate=la)
        self.classifier.compile(optimizer=opt, loss=loss_function, metrics=['accuracy'])

    def train_classifier(self, x_train, y_train, x_test, y_test):
        if self.do_training == True:
            train_history = []
            test_history = []
            for i in range(self.epochs):
                train_history.append(self.classifier.fit(x_train, y_train, epochs=1, batch_size=self.batch_size).history['accuracy'][0])
                test_history.append(self.classifier.evaluate(x_test, y_test, batch_size=self.batch_size)[1])
            if self.store_parameters_after_training == True:
                utility.store_model(self.classifier, self.model_name)
            self.encoder.compile()
            return train_history, test_history
        else:
            self.classifier = utility.load_model(self.model_name)
            self.classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            self.encoder.compile()

    def evaluate(self, x_test, y_test):
        return self.classifier.evaluate(x_test, y_test, verbose=1)

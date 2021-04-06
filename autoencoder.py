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
            self.latent_layer = Dense(600, activation='relu')(layers)
    
    def is_trainable(trainable):
        for layer in self.encoder.layers:
            layer.trainable = trainable
        self.encoder.layers[-1].trainable = True # The latent layer is trainable

    def get_latent_layer(self):
        return self.latent_layer

    def get_input_layer(self):
        return self.input_layer
    
    def get_encoder(self):
        return self.encoder

class Autoencoder:
    def __init__(self, encoder, do_training, store_parameters_after_training, model_name):
        self.do_training = do_training
        self.store_parameters_after_training = store_parameters_after_training
        self.model_name = model_name
        self.encoder = encoder
        if encoder.dataset == "mnist":
            latent_layer = encoder.get_latent_layer()
            layers = Dense(120, activation='relu')(latent_layer)
            layers = Dense(784, activation='sigmoid')(layers)
            self.autoencoder = Model(encoder.get_input_layer(), layers)
            self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        elif encoder.dataset == 'fashion_mnist':
            latent_layer = encoder.get_latent_layer()
            layers = Dense(120, activation='relu')(latent_layer)
            layers = Dense(784, activation='sigmoid')(layers)
            self.autoencoder = Model(encoder.get_input_layer(), layers)
            self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        elif encoder.dataset == 'cifer10':
            self.image_size = 32
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
            self.autoencoder.fit(x_train, x_train, epochs=1, batch_size=100)
            if self.store_parameters_after_training == True:
                utility.store_model(self.autoencoder, self.model_name)
        else:
            self.autoencoder = utility.load_model(self.model_name)
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


class Classifier:
    def __init__(self, encoder, do_training, store_parameters_after_training, model_name):
        self.do_training = do_training
        self.store_parameters_after_training = store_parameters_after_training
        self.model_name = model_name
        if encoder.dataset == 'mnist':
            input_layer = encoder.get_input_layer()
            latent_layer = encoder.get_latent_layer()
            classifier_layer = Dense(10, activation='sigmoid')(latent_layer) 
            self.classifier = Model(input_layer, classifier_layer)
            self.classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        elif encoder.dataset == 'fashion_mnist':
            input_layer = encoder.get_input_layer()
            latent_layer = encoder.get_latent_layer()
            classifier_layer = Dense(10, activation='sigmoid')(latent_layer) 
            self.classifier = Model(input_layer, classifier_layer)
            self.classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        elif encoder.dataset == 'cifer10':
            latent_layer = encoder.get_latent_layer()
            classifier_layer = Flatten()(latent_layer) 
            classifier_layer = Dense(10, activation='sigmoid')(classifier_layer) 
            input_layer = encoder.get_input_layer()
            self.classifier = Model(input_layer, classifier_layer)
            self.classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train_classifier(self, x_train, y_train):
        if self.do_training == True:
            self.classifier.fit(x_train, y_train, epochs=1, batch_size=100)
            if self.store_parameters_after_training == True:
                utility.store_model(self.classifier, self.model_name)
        else:
            self.classifier = utility.load_model(self.model_name)
            self.classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    def evaluate(self, x_test, y_test):
        return self.classifier.evaluate(x_test, y_test, verbose=1)
 
def main():
    dataset = "cifer10"
    (x_train, y_train), (x_test, y_test) = utility.get_dataset(dataset)
    (x_train_D1, y_train_D1), (x_train_D2, y_train_D2) = utility.split_dataset(x_train, y_train)

    autoencoder_do_training = False
    autoencoder_store_model = False
    autoencoder_model_name = 'autoencoder' + str(dataset)
    encoder = Encoder(dataset)
    autoencoder = Autoencoder(encoder, autoencoder_do_training, autoencoder_store_model, autoencoder_model_name)
    autoencoder.train(x_train) 
    autoencoder.show_reconstruction(x_test)
    quit()

    classifier_do_training = True
    classifier_store_model = True
    classifer_model_name = 'auto_classifier' + str(dataset)
    autoencoder_classifier = Classifier(encoder, classifier_do_training, classifier_store_model, classifer_model_name)
    autoencoder_classifier.train_classifier(x_train, y_train)
    loss, acc = autoencoder_classifier.evaluate(x_test, y_test)
    print("Autoencoder classifier loss: " + str(loss) + ". Acc: " + str(acc))

    classifier_do_training = True
    classifier_store_model = True
    classifier_model_name = 'simple_classifier' + str(dataset)
    encoder = Encoder(dataset)
    simple_classifier = Classifier(encoder, classifier_do_training, classifier_store_model, classifier_model_name)
    simple_classifier.train_classifier(x_train, y_train)
    loss, acc = simple_classifier.evaluate(x_test, y_test)
    print("Simple classifier loss: " + str(loss) + ". Acc: " + str(acc))

    # plot_autoencoder_outputs(autoencoder.autoencoder, 5, (image_size, image_size), x_train)



if __name__ == '__main__':
    main()

import tensorflow as tf
import data
import utility
import keras
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, Conv2DTranspose, Flatten, MaxPooling2D, Dropout, Reshape, BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD

def plot_autoencoder_outputs(autoencoder, n, dims, x_test):
    decoded_imgs = autoencoder.predict(x_test)
    # number of example digits to show
    n = 5
    plt.figure(figsize=(10, 4.5))
    for i in range(n):
        # plot original image
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(*dims))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n/2:
            ax.set_title('Original Images')

        # plot reconstruction 
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(*dims))
        plt.show()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n/2:
            ax.set_title('Reconstructed Images')
        plt.show()

class Autoencoder:
    def __init__(self, input_shape, latent_size, encoder_layers, decoder_layers):
        layers = Input(shape=input_shape)
        input_shape = layers
        for i in range(len(encoder_layers)):
            layers = Dense(encoder_layers[i], activation='relu')(layers)

        latent_layer = Dense(latent_size, activation='relu')(layers) # Latent layer

        for i in range(len(decoder_layers)):
            if i == len(decoder_layers) - 1:
                layers = Dense(decoder_layers[i], activation='sigmoid')(layers)
            elif i == 0:
                layers = Dense(decoder_layers[i], activation='relu')(latent_layer)
            else:
                layers = Dense(decoder_layers[i], activation='relu')(layers)

        self.autoencoder = Model(input_shape, layers)
        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

        classifier_layer = Dense(10, activation='sigmoid')(latent_layer) 
        self.classifier = Model(input_shape, classifier_layer)
        self.classifier.compile(optimizer='adam', loss='binary_crossentropy')

    def train_autoencoder(self, x_train):
        self.autoencoder.fit(x_train, x_train, epochs=1)

    def train_classifier(self, x_train, y_train):
        self.classifier.fit(x_train, y_train, epochs=10)

    def evaluate(self, x_test, y_test):
        return self.classifier.evaluate(x_test, y_test)

    
def autotest():
    dataset = "fashion_mnist"
    if dataset == "mnist":
        image_size = 28
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        y_test = to_categorical(y_test, 10)
        y_train = to_categorical(y_train, 10)

        # Unlabled
        x_train_D1 = x_train[0 : len(x_train) - len(x_train) // 30]
        y_train_D1 = None
        # Labled
        x_train_D2 = x_train[len(x_train) - len(x_train) // 30 :]
        y_train_D2 = y_train[len(y_train) - len(y_train) // 30 :]
    
        latent_units_size = 1000
        autoencoder = Autoencoder((image_size * image_size), latent_units_size, [784, 120], [120, 784])
        autoencoder.train_autoencoder(x_train_D1)
        autoencoder.train_classifier(x_train_D2, y_train_D2)

        classifier = Autoencoder((image_size * image_size), latent_units_size, [784, 120], [120, 784])
        classifier.train_classifier(x_train_D2, y_train_D2)

        result1 = autoencoder.evaluate(x_test, y_test)
        result2 = classifier.evaluate(x_test, y_test)
        print("Loss autoencoder: " + str(result1))
        print("Loss classifier: " + str(result2))
        #plot_autoencoder_outputs(autoencoder.autoencoder, 5, (image_size, image_size), x_train)
    elif dataset == "fashion_mnist":
        image_size = 28
        latent_units_size = 100
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        y_test = to_categorical(y_test, 10)
        y_train = to_categorical(y_train, 10)

        # Unlabled
        x_train_D1 = x_train[0 : len(x_train) - len(x_train) // 30]
        y_train_D1 = None
        # Labled
        x_train_D2 = x_train[len(x_train) - len(x_train) // 30 :]
        y_train_D2 = y_train[len(y_train) - len(y_train) // 30 :]
    
        autoencoder = Autoencoder((image_size * image_size), latent_units_size, [784, 120], [120, 784])
        autoencoder.train_autoencoder(x_train_D1)
        autoencoder.train_classifier(x_train_D2, y_train_D2)

        classifier = Autoencoder((image_size * image_size), latent_units_size, [784, 120], [120, 784])
        classifier.train_classifier(x_train_D2, y_train_D2)

        result1 = autoencoder.evaluate(x_test, y_test)
        result2 = classifier.evaluate(x_test, y_test)
        print("Loss autoencoder: " + str(result1))
        print("Loss classifier: " + str(result2))
        
        #plot_autoencoder_outputs(autoencoder.autoencoder, 5, (image_size, image_size), x_train)
    else:
        image_size = 32
        latent_units_size = 20
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        y_test = to_categorical(y_test, 10)
        y_train = to_categorical(y_train, 10)

        # Unlabled
        x_train_D1 = x_train[0 : len(x_train) - len(x_train) // 5]
        y_train_D1 = None
        # Labled
        x_train_D2 = x_train[len(x_train) - len(x_train) // 5 :]
        y_train_D2 = y_train[len(y_train) - len(y_train) // 5 :]

        input_shape = Input(shape=(32, 32, 3))
        layers = input_shape
        layers = Conv2D(filters=16, kernel_size=(6, 6), activation='relu')(layers)
        layers = Conv2D(filters=8, kernel_size=(3, 3), activation='relu')(layers)
        layers = MaxPooling2D((2, 2))(layers)
        layers = Conv2D(filters=8, kernel_size=(3, 3), activation='relu')(layers)
        layers = Conv2D(filters=4, kernel_size=(3, 3), activation='relu')(layers)
        layers = MaxPooling2D((2, 2))(layers)
        layers = Flatten()(layers)
        layers = Dense(1200, activation='relu')(layers)
        latent_output = Dense(10, activation='sigmoid')(layers)
        layers = Reshape((20, 20, 3))(layers)
        layers = Conv2DTranspose(filters=4, kernel_size=(3, 3), activation='relu')(layers)
        layers = Conv2DTranspose(filters=8, kernel_size=(3, 3), activation='relu')(layers)
        layers = MaxPooling2D((1, 1))(layers)
        layers = Conv2DTranspose(filters=16, kernel_size=(4, 4), activation='relu')(layers)
        layers = Conv2DTranspose(filters=3, kernel_size=(6, 6), activation='sigmoid')(layers)
        layers = Reshape((32, 32, 3))(layers)

        opt = SGD(lr=0.1, momentum=0.0)
        autoencoder = Model(input_shape, layers)
        autoencoder.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        autoencoder.fit(x_train_D1, x_train_D1, epochs=1)

        classifier1 = Model(input_shape, latent_output)
        classifier1.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        classifier1.fit(x_train_D2, y_train_D2)

        result1 = classifier1.evaluate(x_test, y_test, verbose=0)
        print("Autoencoder:")
        print(result1)

        #autoencoder.fit(x_train, y_train, epochs=1)
        #plot_autoencoder_outputs(autoencoder, 5, (image_size, image_size, 3), x_train)

def main():
    autotest()
    quit()
    image_size = 32
    latent_units_size = 20
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_test = to_categorical(y_test, 10)
    y_train = to_categorical(y_train, 10)

    # Unlabled
    x_train_D1 = x_train[0 : len(x_train) - len(x_train) // 5]
    y_train_D1 = None
    # Labled
    x_train_D2 = x_train[len(x_train) - len(x_train) // 5 :]
    y_train_D2 = y_train[len(y_train) - len(y_train) // 5 :]

    opt = SGD(lr=0.1, momentum=0.0)
    input_shape = Input(shape=(32, 32, 3))
    layers = input_shape
    layers = Conv2D(filters=16, kernel_size=(6, 6), activation='relu')(layers)
    layers = Conv2D(filters=8, kernel_size=(3, 3), activation='relu')(layers)
    layers = MaxPooling2D((2, 2))(layers)
    layers = Conv2D(filters=8, kernel_size=(3, 3), activation='relu')(layers)
    layers = Conv2D(filters=4, kernel_size=(3, 3), activation='relu')(layers)
    layers = MaxPooling2D((2, 2))(layers)
    layers = Flatten()(layers)
    layers = Dense(1200, activation='relu')(layers)
    latent_output = Dense(10, activation='sigmoid')(layers)

    classifier2 = Model(input_shape, latent_output)
    classifier2.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    classifier2.fit(x_train_D2, y_train_D2)

    result2 = classifier2.evaluate(x_test, y_test, verbose=0)
    print("Classifier:")
    print(result2)

if __name__ == '__main__':
    main()

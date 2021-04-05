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
        self.autoencoder.fit(x_train, x_train, epochs=1, batch_size=100)

    def train_classifier(self, x_train, y_train):
        self.classifier.fit(x_train, y_train, epochs=1, batch_size=100)

    def evaluate(self, x_test, y_test):
        return self.classifier.evaluate(x_test, y_test)


class Encoder:
    def __init__(self):
        image_size = 28
        input_shape = (image_size * image_size)
        autoencoder = Autoencoder((image_size * image_size), latent_units_size, [784, 120], [120, 784])

        layers = Input(shape=input_shape)
        input_shape = layers
        layers = Dense(784, activation='relu')(layers)
        layers = Dense(120, activation='relu')(layers)
        self.latent_layer = Dense(1000, activation='relu')(layers) # Latent layer
        self.encoder = Model(input_shape, layers)

    
    def is_trainable(trainable):
        for layer in self.encoder.layers:
            layer.trainable = trainable
        self.encoder.layers[-1].trainable = True # The latent layer is trainable

    def get_latent_layer(self):
        return self.latent_layer

class Decoder:
    def __init__(self, encoder):
        latent_layer = encoder.get_latent_layer()
        layers = Dense(120, activation='relu')(layers)
        layers = Dense(784, activation='sigmoid')(layers)
        self.decoder = Model(latent_layer, layers)
        self.decoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def get_decoder(self):
        return self.decoder

    def train(self, x_test):
        autoencoder.fit(x_test, x_test, epochs=1, batch_size=50)

def autotest():
    dataset = "mnist"
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
        x_train_D1 = x_train[0 : len(x_train) - len(x_train) // 20]
        y_train_D1 = None
        # Labled
        x_train_D2 = x_train[len(x_train) - len(x_train) // 20 :]
        y_train_D2 = y_train[len(y_train) - len(y_train) // 20 :]
    
        latent_units_size = 1000
        autoencoder = Autoencoder((image_size * image_size), latent_units_size, [784, 120], [120, 784])
        autoencoder.train_autoencoder(x_train_D1)
        autoencoder.train_autoencoder(x_train_D1)
        autoencoder.train_classifier(x_train_D2, y_train_D2)

        classifier = Autoencoder((image_size * image_size), latent_units_size, [784, 120], [120, 784])
        classifier.train_classifier(x_train_D2, y_train_D2)

        result1 = autoencoder.classifier.evaluate(x_test, y_test, verbose=0)
        result2 = classifier.classifier.evaluate(x_test, y_test, verbose=0)
        print("Loss autoencoder:")
        print(result1)
        print("Loss classifier")
        print(result2)
        quit()
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
        x_train_D1 = x_train[0 : len(x_train) - len(x_train) // 30]
        y_train_D1 = None
        # Labled
        x_train_D2 = x_train[len(x_train) - len(x_train) // 20 :]
        y_train_D2 = y_train[len(y_train) - len(y_train) // 20 :]

        input_shape = Input(shape=(32, 32, 3))
        layers = input_shape
        layers = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(layers)
        layers = BatchNormalization()(layers)
        layers = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(layers)
        layers = BatchNormalization()(layers)
        layers = Dense(400, activation='relu')(layers)
        latent_layer = layers
        layers = UpSampling2D()(layers)
        layers = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(layers)
        layers = BatchNormalization()(layers)
        layers = Conv2D(3,  kernel_size=1, strides=1, padding='same', activation='sigmoid')(layers)

        autoencoder = Model(input_shape, layers)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        #autoencoder.fit(x_train_D1, x_train_D1, epochs=1, batch_size=50)
        #utility.store_model(autoencoder, 'autoencoder')
        autoencoder = utility.load_model('autoencoder')
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        classifier_layer = Flatten()(latent_layer) 
        classifier_layer = Dense(10, activation='sigmoid')(classifier_layer) 
        classifier = Model(input_shape, classifier_layer)

        #for layer in autoencoder.layers:
        #    layer.trainable = False
        opt = keras.optimizers.Adam(learning_rate=0.0005) 
        classifier.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        classifier.fit(x_train_D2, y_train_D2, batch_size=5, epochs=1)
        result1 = classifier.evaluate(x_test, y_test, verbose=0)
        print("Autoencoder loss and accurarcy:")
        print(result1)
        #plot_autoencoder_outputs(autoencoder, 5, (image_size, image_size, 3), x_train)

def main():
    autotest()
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_test = to_categorical(y_test, 10)
    y_train = to_categorical(y_train, 10)

    # Unlabled
    x_train_D1 = x_train[0 : len(x_train) - len(x_train) // 30]
    y_train_D1 = None
    # Labled
    x_train_D2 = x_train[len(x_train) - len(x_train) // 20 :]
    y_train_D2 = y_train[len(y_train) - len(y_train) // 20 :]

    input_shape = Input(shape=(32, 32, 3))
    layers = input_shape
    layers = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(layers)
    layers = BatchNormalization()(layers)
    layers = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(layers)
    layers = BatchNormalization()(layers)
    layers = Dense(400, activation='relu')(layers)
    layers = Flatten()(layers) 
    layers = Dense(10, activation='sigmoid')(layers) 
    classifier = Model(input_shape, layers)
    opt = keras.optimizers.Adam(learning_rate=0.001)
    classifier.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    classifier.fit(x_train_D2, y_train_D2, batch_size=5, epochs=1)

    result2 = classifier.evaluate(x_test, y_test, verbose=0)
    print("Classifier loss and accurarcy:")
    print(result2)

if __name__ == '__main__':
    main()

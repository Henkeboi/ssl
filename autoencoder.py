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

class Encoder:
    def __init__(self, la, conv_layers, dense_layers):
        self.model = tf.keras.models.Sequential()
        for i in range(len(conv_layers)):
            filters = conv_layers[i].filters
            kernel_size = conv_layers[i].kernel_size
            input_shape = conv_layers[i].input_shape
            af = conv_layers[i].af
            if input_shape == None:
                self.model.add(tf.keras.layers.Conv2D(filters, kernel_size, activation=af))
            else:
                self.model.add(tf.keras.layers.Conv2D(filters, kernel_size, input_shape=input_shape, activation=af))
        self.model.add(tf.keras.layers.Flatten())
        for i in range(len(dense_layers)):
            neurons = dense_layers[i].neurons
            af = dense_layers[i].af
            self.model.add(tf.keras.layers.Dense(neurons, activation=af))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy') 

    def train(self, x_train, y_train, epochs):
        # Batch, height, width, channels
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        y_train = to_categorical(y_train, 10)
        self.model.fit(x_train, y_train, epochs=epochs)
            
class Decoder:
    def __init__(self, la, conv_layers, dense_layers):
        self.encoder = Encoder(la, conv_layers, dense_layers)
        self.model = tf.keras.models.Sequential()
                
        for i in range(len(dense_layers)):
            neurons = dense_layers[i].neurons
            af = dense_layers[i].af
            self.model.add(tf.keras.layers.Dense(neurons, activation=af))

        for i in range(len(conv_layers)):
            filters = conv_layers[i].filters
            kernel_size = conv_layers[i].kernel_size
            input_shape = conv_layers[i].input_shape
            self.model.add(tf.keras.layers.Conv2DTranspose(filters, kernel_size, activation=af))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')

    def train(self, x_train, y_train, epochs):
        pass

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

        self.encoder = Model(input_shape, latent_layer)
        self.encoder.compile(optimizer='adam', loss='binary_crossentropy')

    def train(self, x_train):
        self.autoencoder.fit(x_train, x_train, epochs=3)
    
def autotest():
    dataset = "cifar"
    if dataset == "mnist":
        image_size = 28
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        latent_units_size = 20
        autoencoder = Autoencoder((image_size * image_size), latent_units_size, [784, 120], [120, 784])
        autoencoder.train(x_train)
        plot_autoencoder_outputs(autoencoder.autoencoder, 5, (image_size, image_size), x_train)
    elif dataset == "fashion_mnist":
        image_size = 28
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        autoencoder = Autoencoder((image_size * image_size), latent_units_size, [784, 120], [120, 784])
        autoencoder.train(x_train)
        plot_autoencoder_outputs(autoencoder.autoencoder, 5, (image_size, image_size), x_train)
    else:
        image_size = 32
        latent_units_size = 20
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        y_test = to_categorical(y_test, 10)
        y_train = to_categorical(y_train, 10)
        x_train = x_train[0 : len(x_train) - len(x_train) // 3]
        y_train = y_train[0 : len(y_train) - len(y_train) // 3]

        # Unlabled
        x_train_D1 = x_train[0 : len(x_train) - len(x_train) // 3]
        y_train_D1 = None
        # Labled
        x_train_D2 = x_train[len(x_train) - len(x_train) // 3 :]
        y_train_D2 = y_train[len(y_train) - len(y_train) // 3 :]

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
        layers = MaxPooling2D((1, 1))(layers)
        layers = Reshape((32, 32, 3))(layers)

        opt = SGD(lr=0.1, momentum=0.1)
        autoencoder = Model(input_shape, layers)
        autoencoder.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        autoencoder.fit(x_train_D1, x_train_D1, epochs=1)

        classifier1 = Model(input_shape, latent_output)
        classifier1.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        classifier1.fit(x_train_D2, y_train_D2)

        
        result1 = classifier1.evaluate(x_test, y_test, verbose=0)
        print(result1)

        #autoencoder.fit(x_train, y_train, epochs=1)
        #plot_autoencoder_outputs(autoencoder, 5, (image_size, image_size, 3), x_train)

def main():
    autotest()
    image_size = 32
    latent_units_size = 20
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_test = to_categorical(y_test, 10)
    y_train = to_categorical(y_train, 10)
    x_train = x_train[0 : len(x_train) - len(x_train) // 3]
    y_train = y_train[0 : len(y_train) - len(y_train) // 3]

    # Unlabled
    x_train_D1 = x_train[0 : len(x_train) - len(x_train) // 3]
    y_train_D1 = None
    # Labled
    x_train_D2 = x_train[len(x_train) - len(x_train) // 3 :]
    y_train_D2 = y_train[len(y_train) - len(y_train) // 3 :]


    opt = SGD(lr=0.1, momentum=0.1)
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
    print(result2)

    quit()

    (x_train, y_train), (x_validation, y_validation) = data.load_mnist()

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    input_shape = (28, 28, 1)
    inputs = Input(shape=input_shape)
    y = Conv2D(filters=1, kernel_size=(2, 2), activation='relu')(inputs)
    y = MaxPooling2D()(y)
    y = Conv2D(filters=1, kernel_size=1, activation='relu')(y)
    y = Flatten()(y)
    y = Dropout(0.1)(y)
    outputs = Dense(28 * 28, activation='sigmoid')(y)
    #y = Reshape(target_shape=(10, 10, 1))(y)
    #y = Conv2DTranspose(filters=1 , kernel_size=(1, 1), activation='relu')(y)
    #outputs = Conv2DTranspose(filters=1 , kernel_size=(1, 1), activation='softmax')(y)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(x_train, x_train)
    #plot_autoencoder_outputs(model, 5, (28, 28), x_train)

    quit()

    la = 0.1
    conv_input_shapes = []
    conv_filters = []
    conv_kernel_sizes = []
    conv_af = []

    conv_input_shapes.append((28, 28, 1))
    conv_filters.append(5)
    conv_kernel_sizes.append((2, 2))
    conv_af.append(tf.nn.relu)

    conv_input_shapes.append(None)
    conv_filters.append(2)
    conv_kernel_sizes.append((2, 2))
    conv_af.append(tf.nn.relu)

    conv_layers = []
    for i in range(len(conv_input_shapes)):
        conv_layers.append(utility.ConvConfig(conv_input_shapes[i], conv_filters[i], conv_kernel_sizes[i], conv_af[i])) 

    dense_neurons = []
    dense_af = []

    dense_neurons.append(10)
    dense_af.append(tf.nn.softmax)

    dense_layers = []
    for i in range(len(dense_neurons)):
        dense_layers.append(utility.DenseConfig(dense_neurons[i], dense_af[i]))

    encoder = Encoder(la, conv_layers, dense_layers)
    dense_layers = dense_layers[::-1]
    conv_layers = conv_layers[::-1]
    decoder = Decoder(la, conv_layers, dense_layers)

    #encoder.train(x_train, y_train, 3)

if __name__ == '__main__':
    main()

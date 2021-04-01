import tensorflow as tf
import data
import utility
import keras
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, Conv2DTranspose, Flatten, MaxPooling2D, Dropout
import numpy as np
import matplotlib.pyplot as plt

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
        plt.gray()
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


def autotest():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    input_size = 784
    hidden_size = 128
    code_size = 32

    input_img = Input(shape=(input_size,))
    hidden_1 = Dense(hidden_size, activation='relu')(input_img)
    code = Dense(code_size, activation='relu')(hidden_1)
    hidden_2 = Dense(hidden_size, activation='relu')(code)
    output_img = Dense(input_size, activation='sigmoid')(hidden_2)

    code1 = Dense(input_size, activation='relu')(hidden_1)


    autoencoder = Model(input_img, output_img)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(x_train, x_train, epochs=3)

    #encoder = Model(input_img, code1)
    #encoder.compile(optimizer='adam', loss='binary_crossentropy')
    #encoder.fit(x_train, x_train)

    plot_autoencoder_outputs(autoencoder, 5, (28, 28), x_test)


def main():
    autotest()
    quit()
    (x_train, y_train), (x_validation, y_validation) = data.load_mnist()

    input_shape = (28, 28, 1)
    inputs = Input(shape=input_shape)
    y = Conv2D(filters=1, kernel_size=(3, 3), activation='relu')(inputs)
    y = MaxPooling2D()(y)
    y = Conv2D(filters=1, kernel_size=1, activation='relu')(y)
    y = Flatten()(y)
    y = Dropout(0.01)(y)
    y = Dense(100, activation='softmax')(y)
    outputs = Dense(10, activation='softmax')(y)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    y_train = to_categorical(y_train, 10)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(x_train, y_train)

    quit()
    #la = 0.1
    #conv_input_shapes = []
    #conv_filters = []
    #conv_kernel_sizes = []
    #conv_af = []

    #conv_input_shapes.append((28, 28, 1))
    #conv_filters.append(5)
    #conv_kernel_sizes.append((2, 2))
    #conv_af.append(tf.nn.relu)

    #conv_input_shapes.append(None)
    #conv_filters.append(2)
    #conv_kernel_sizes.append((2, 2))
    #conv_af.append(tf.nn.relu)

    #conv_layers = []
    #for i in range(len(conv_input_shapes)):
    #    conv_layers.append(utility.ConvConfig(conv_input_shapes[i], conv_filters[i], conv_kernel_sizes[i], conv_af[i])) 

    #dense_neurons = []
    #dense_af = []

    #dense_neurons.append(10)
    #dense_af.append(tf.nn.softmax)

    #dense_layers = []
    #for i in range(len(dense_neurons)):
    #    dense_layers.append(utility.DenseConfig(dense_neurons[i], dense_af[i]))

    #encoder = Encoder(la, conv_layers, dense_layers)
    #dense_layers = dense_layers[::-1]
    #conv_layers = conv_layers[::-1]
    #decoder = Decoder(la, conv_layers, dense_layers)

    #encoder.train(x_train, y_train, 3)

if __name__ == '__main__':
    main()

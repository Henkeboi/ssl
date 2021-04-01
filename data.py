import numpy as np
import keras
import matplotlib.pyplot as plt
import tensorflow as tf

def load_mnist():
    (x_train, y_train), (x_validation, y_validation) = keras.datasets.mnist.load_data()
    return (x_train, y_train), (x_validation, y_validation)



def show(data):
    fig = plt.figure()
    plt.imshow(data)
    plt.show()

def normalize(data):
    data = tf.keras.utils.normalize(data, axis=1)
    return data


def main():
    (x_train, y_train), (x_validation, y_validation) = load_mnist()
    x_train = normalize(x_train) 
    x_validation = normalize(x_validation) 

if __name__ == '__main__':
    main()

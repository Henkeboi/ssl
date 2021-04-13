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
from autoencoder import Autoencoder
import glob
from PIL import Image
from config import Config

def main():
    config = Config()
    config_dict = config.get_config()

    dataset = config_dict['dataset']
    la_autoencoder = config_dict[dataset + '_la_autoencoder']
    la_autoencoder_classifier = config_dict[dataset + '_la_autoencoder_classifier']
    la_simple_classifier = config_dict[dataset + '_la_simple_classifier']
    loss_function_autoencoder = config_dict['loss_function_autoencoder']
    loss_function_classifier = config_dict['loss_function_classifier']
    optimizer_autoencoder = config_dict['optimizer_autoencoder']
    optimizer_classifier = config_dict['optimizer_classifier']
    latent_size = config_dict[dataset + '_latent_size']
    autoencoder_epochs = config_dict[dataset + '_autoencoder_epochs']
    classifier_epochs = config_dict[dataset + '_classifier_epochs']
    freeze_encoder = config_dict['freeze']
    D1D2_ratio = config_dict[dataset + '_D1D2_fraction']
    D2_training_ratio = config_dict[dataset + '_D2_training_fraction']
    num_reconstructions = config_dict['num_reconstructions']
    plot_tsne = config_dict['plot_tSNE']
    num_images = 250
    plot_learning = config_dict['plot_learning']

    x_data, y_data = utility.get_dataset(dataset)
    x_train_D1, (x_train_D2, y_train_D2), (x_test_D2, y_test_D2) = utility.split_dataset(x_data, y_data, D1D2_ratio, D2_training_ratio)

    # Autoencoder
    encoder = Encoder(dataset, latent_size)
    if plot_tsne == 1:
        encoder.plot_tSNE(num_images, 'Stage 1')

    autoencoder_do_training = True
    autoencoder_store_model = True
    autoencoder_model_name = 'autoencoder' + str(dataset)
    autoencoder = Autoencoder(encoder, freeze_encoder, la_autoencoder, loss_function_autoencoder, optimizer_autoencoder, autoencoder_epochs, autoencoder_do_training, autoencoder_store_model, autoencoder_model_name)


    if plot_learning and autoencoder_do_training:
        x_train, x_test = autoencoder.train(x_train_D1, x_test_D2)
        fig, ax = plt.subplots()
        fig.suptitle('Autoencoder loss')
        plt.plot(x_train, '-r', label='Training loss')
        plt.plot(x_test, '-b', label='Validation loss')
        plt.xlabel('Batches')
        plt.ylabel('Loss')
        leg = ax.legend()
        plt.show(block=False)
    else:
        autoencoder.train(x_train_D1, x_test_D2) 
    
    for i in range(num_reconstructions):
        autoencoder.show_reconstruction(x_test_D2[i])

    if plot_tsne == 1:
        autoencoder.get_encoder().plot_tSNE(num_images, 'Stage 2')

    auto_classifier_do_training = True
    auto_classifier_store_model = True
    auto_classifer_model_name = 'auto_classifier' + str(dataset)
    autoencoder_classifier = Classifier(encoder, la_autoencoder_classifier, loss_function_classifier, optimizer_classifier, classifier_epochs, auto_classifier_do_training, auto_classifier_store_model, auto_classifer_model_name)

    simple_encoder = Encoder(dataset, latent_size)
    simple_classifier_do_training = True
    simple_classifier_store_model = True
    simple_classifier_model_name = 'simple_classifier' + str(dataset)
    simple_classifier = Classifier(simple_encoder, la_simple_classifier, loss_function_classifier, optimizer_classifier, classifier_epochs, simple_classifier_do_training, simple_classifier_store_model, simple_classifier_model_name)



    if plot_learning and auto_classifier_do_training and simple_classifier_do_training:
        auto_train_acc, auto_test_acc = autoencoder_classifier.train_classifier(x_train_D2, y_train_D2, x_test_D2, y_test_D2)
        simple_train_acc, simple_test_acc = simple_classifier.train_classifier(x_train_D2, y_train_D2, x_test_D2, y_test_D2)
        fig, ax = plt.subplots()
        fig.suptitle('Classifier accuracy')
        plt.plot(auto_train_acc, '-r', label='Auto training accuracy')
        plt.plot(auto_test_acc, '-b', label='Auto validation accuracy')
        plt.plot(simple_train_acc, '-g', label='Simple training accuracy')
        plt.plot(simple_test_acc, '-y', label='Simple validation accuracy')
        plt.xlabel('Batches')
        plt.ylabel('Accuracy')
        leg = ax.legend()
        plt.show(block=False)
    else:
        autoencoder_classifier.train_classifier(x_train_D2, y_train_D2, x_test_D2, y_test_D2)
        simple_classifier.train_classifier(x_train_D2, y_train_D2, x_test_D2, y_test_D2)

    if plot_tsne == 1:
        autoencoder.get_encoder().plot_tSNE(num_images, 'Stage 3')
    plt.show()

   

if __name__ == '__main__':
    main()

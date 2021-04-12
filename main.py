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

    x_data, y_data = utility.get_dataset(dataset)
    x_train_D1, (x_train_D2, y_train_D2), (x_test_D2, y_test_D2) = utility.split_dataset(x_data, y_data, D1D2_ratio, D2_training_ratio)


    # Autoencoder
    encoder = Encoder(dataset, latent_size)
    autoencoder_do_training = False
    autoencoder_store_model = False
    autoencoder_model_name = 'autoencoder' + str(dataset)
    autoencoder = Autoencoder(encoder, freeze_encoder, la_autoencoder, loss_function_autoencoder, optimizer_autoencoder, autoencoder_epochs, autoencoder_do_training, autoencoder_store_model, autoencoder_model_name)
    autoencoder.train(x_train_D1) 
    #autoencoder.show_reconstruction(x_train_D2)

    classifier_do_training = True
    classifier_store_model = True
    classifer_model_name = 'auto_classifier' + str(dataset)
    autoencoder_classifier = Classifier(encoder, la_autoencoder_classifier, loss_function_classifier, optimizer_classifier, classifier_epochs, classifier_do_training, classifier_store_model, classifer_model_name)
    autoencoder_classifier.train_classifier(x_train_D2, y_train_D2)
    loss, acc = autoencoder_classifier.evaluate(x_test_D2, y_test_D2)
    print("Autoencoder classifier loss: " + str(loss) + ". Acc: " + str(acc))
    #encoder.plot_tSNE()

    simple_encoder = Encoder(dataset, latent_size)
    classifier_do_training = True
    classifier_store_model = True
    classifier_model_name = 'simple_classifier' + str(dataset)
    simple_classifier = Classifier(simple_encoder, la_simple_classifier, loss_function_classifier, optimizer_classifier, classifier_epochs, classifier_do_training, classifier_store_model, classifier_model_name)
    simple_classifier.train_classifier(x_train_D2, y_train_D2)
    loss, acc = simple_classifier.evaluate(x_test_D2, y_test_D2)

    print("Simple classifier loss: " + str(loss) + ". Acc: " + str(acc))

if __name__ == '__main__':
    main()

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

import os

num_skipped = 0
for folder_name in ("cats", "dogs"):
    folder_path = os.path.join("data/pets", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            os.remove(fpath)

def main():
    dataset = "mnist"
    (x_train, y_train), (x_test, y_test) = utility.get_dataset(dataset)
    (x_train_D1, y_train_D1), (x_train_D2, y_train_D2) = utility.split_dataset(x_train, y_train)
    x_train = None
    y_train = None

    encoder = Encoder(dataset)
    autoencoder_do_training = False
    autoencoder_store_model = True
    autoencoder_model_name = 'autoencoder' + str(dataset)
    autoencoder = Autoencoder(encoder, autoencoder_do_training, autoencoder_store_model, autoencoder_model_name)
    autoencoder.train(x_train_D1) 
    #autoencoder.show_reconstruction(x_train_D2)
    
    classifier_do_training = False
    classifier_store_model = True
    classifer_model_name = 'auto_classifier' + str(dataset)
    autoencoder_classifier = Classifier(encoder, classifier_do_training, classifier_store_model, classifer_model_name)
    autoencoder_classifier.train_classifier(x_train_D2, y_train_D2)
    loss, acc = autoencoder_classifier.evaluate(x_test, y_test)
    print("Autoencoder classifier loss: " + str(loss) + ". Acc: " + str(acc))

    encoder = Encoder(dataset)
    classifier_do_training = True
    classifier_store_model = True
    classifier_model_name = 'simple_classifier' + str(dataset)
    simple_classifier = Classifier(encoder, classifier_do_training, classifier_store_model, classifier_model_name)
    simple_classifier.train_classifier(x_train_D2, y_train_D2)
    loss, acc = simple_classifier.evaluate(x_test, y_test)

    (x_train, y_train), (x_test, y_test) = utility.get_dataset(dataset)
    encoder.plot_tSNE(x_train, y_train)
    print("Simple classifier loss: " + str(loss) + ". Acc: " + str(acc))



if __name__ == '__main__':
    main()

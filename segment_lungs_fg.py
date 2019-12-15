import argparse
import cv2
import keras.callbacks as kcallbacks
import keras.layers as klayers
import keras.optimizers as koptims
import keras.losses as klosses
import math
import numpy as np
import os
import random
import sys
import warnings
from typing import Callable

import tensorflow as tf

from datetime import datetime
from enum import Enum
from keras.models import Model
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import set_random_seed


# HYPER PARAMETERS
IMG_SHAPE = (256, 256, 1)
IMG_TRAIN_FOLDER = 'img_train'
IMG_VAL_FOLDER = 'img_val'
IMG_TEST_FOLDER = 'img_test'
IMG_NORMALIZATION = True
BINARIZATION_THRESHOLD_IMAGE = 127
MASK_TRAIN_FOLDER = 'mask_train'
MASK_VAL_FOLDER = 'mask_val'
MASK_TEST_FOLDER = 'mask_test'
MASK_NORMALIZATION = True
BINARIZATION_THRESHOLD_NORM = 0.5
MASK_BALANCING = True

FILES_TRAIN_SIZE = 0.8
FILES_VAL_SIZE = 0.2
FILES_SHUFFLE = True

IMG_EXTENSIONS = ['.png', '.PNG', '.jpg', '.JPG']
WEIGHTS_EXTENSIONS = ['.h5', '.hdf5']

MODEL_PLOT_SCHEME = False

TRAIN_LEARNING_RATE = 0.00000005
TRAIN_LOSS_FUNCTION = klosses.mean_squared_logarithmic_error
TRAIN_BATCH_SIZE = 4
TRAIN_EPOCHS = 500
TRAIN_SHUFFLE = True
TRAIN_VERBOSITY = 1
TRAIN_INTER_RESULTS = True
TRAIN_INTER_RESULTS_NUM = 10

TEST_VERBOSITY = 1
TEST_THRESHOLD = 0.7
TEST_OUTPUT_FOLDER = 'test_output'
TEST_TRUE_REGION_BGR_COLOR = (0, 255, 0)
TEST_PRED_REGION_BGR_COLOR = (0, 0, 255)


class ScriptMode(Enum):
    TRAIN = 'train'
    TEST = 'test'


class ModelType(Enum):
    U_NET = 'unet'
    FG_SEG_NET = 'fgsegnet'


def mask_weighted_loss(loss_func: Callable[[tf.Tensor, tf.Tensor], float], class_weights: dict):
    """
    Creates a loss function with weighted parts calculated for
        zeros component of an input true tensor and for greater-than-zeros component.
    :param loss_func: original loss function.
    :param class_weights: dict of class weights for
        zeros component (key = 0), and for ones component (key = 1).
    :return: weighted loss function.
    """
    if (0 not in class_weights) or (1 not in class_weights):
        raise ValueError('Input class_weights dict must contain the following keys: 0, 1.')

    def loss(true: tf.Tensor, pred: tf.Tensor):
        zeros = tf.zeros_like(true)

        # Create masks for zeros and greater-than-zeros part of true tensor
        zeros_mask = tf.equal(true, zeros)
        ones_mask = tf.greater(true, zeros)

        # Create flatten tensors containing elements after masking
        zero_masked_true = tf.boolean_mask(true, zeros_mask)
        zero_masked_pred = tf.boolean_mask(pred, zeros_mask)
        ones_masked_true = tf.boolean_mask(true, ones_mask)
        ones_masked_pred = tf.boolean_mask(pred, ones_mask)

        # Calculate weighted loss values
        zero_class_weight = float(class_weights[0])
        ones_class_weight = float(class_weights[1])
        zero_loss = zero_class_weight * loss_func(zero_masked_true, zero_masked_pred)
        ones_loss = ones_class_weight * loss_func(ones_masked_true, ones_masked_pred)

        return zero_loss + ones_loss

    return loss


class ModelManager:
    def __init__(self, input_shape: tuple):
        self.__input_shape = input_shape

        self.__unet_plot_name = 'U-net.png'

    def __get_unet(self, weights_path: str=None, plot_scheme: bool=False):
        """
        Creates a model with U-Net architecture.
        :param weights_path: path to the model weights file.
        :param plot_scheme: whether to create the plot of the model architecture.
        :return: Keras model.
        """
        inputs = klayers.Input(self.__input_shape)
        conv1 = klayers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = klayers.ReLU()(conv1)
        conv1 = klayers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = klayers.ReLU()(conv1)
        pool1 = klayers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = klayers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = klayers.ReLU()(conv2)
        conv2 = klayers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = klayers.ReLU()(conv2)
        pool2 = klayers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = klayers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = klayers.ReLU()(conv3)
        conv3 = klayers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = klayers.ReLU()(conv3)
        pool3 = klayers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = klayers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = klayers.ReLU()(conv4)
        conv4 = klayers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = klayers.ReLU()(conv4)
        drop4 = klayers.Dropout(0.5)(conv4)
        pool4 = klayers.MaxPooling2D(pool_size=(2, 2))(drop4)

        # Bottleneck
        conv5 = klayers.Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = klayers.ReLU()(conv5)
        conv5 = klayers.Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = klayers.ReLU()(conv5)
        drop5 = klayers.Dropout(0.5)(conv5)

        up6 = klayers.UpSampling2D(size=(2, 2))(drop5)
        up6 = klayers.Conv2D(512, 2, padding='same', kernel_initializer='he_normal')(up6)
        up6 = klayers.PReLU()(up6)
        merge6 = klayers.concatenate([drop4, up6], axis=3)
        conv6 = klayers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = klayers.ReLU()(conv6)
        conv6 = klayers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = klayers.ReLU()(conv6)

        up7 = klayers.UpSampling2D(size=(2, 2))(conv6)
        up7 = klayers.Conv2D(256, 2, padding='same', kernel_initializer='he_normal')(up7)
        up7 = klayers.ReLU()(up7)
        merge7 = klayers.concatenate([conv3, up7], axis=3)
        conv7 = klayers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = klayers.ReLU()(conv7)
        conv7 = klayers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = klayers.ReLU()(conv7)

        up8 = klayers.UpSampling2D(size=(2, 2))(conv7)
        up8 = klayers.Conv2D(128, 2, padding='same', kernel_initializer='he_normal')(up8)
        up8 = klayers.ReLU()(up8)
        merge8 = klayers.concatenate([conv2, up8], axis=3)
        conv8 = klayers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = klayers.ReLU()(conv8)
        conv8 = klayers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = klayers.ReLU()(conv8)

        up9 = klayers.UpSampling2D(size=(2, 2))(conv8)
        up9 = klayers.Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(up9)
        up9 = klayers.ReLU()(up9)
        merge9 = klayers.concatenate([conv1, up9], axis=3)
        conv9 = klayers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = klayers.ReLU()(conv9)
        conv9 = klayers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = klayers.ReLU()(conv9)
        conv9 = klayers.Conv2D(2, 3, padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = klayers.ReLU()(conv9)

        outputs = klayers.Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs, outputs, name='u-net')
        if plot_scheme:
            plot_model(model, to_file='U-net.png', show_shapes=True)

        if weights_path is not None:
            try:
                model.load_weights(os.path.abspath(weights_path))
            except ValueError:
                raise ValueError('U-net weights does not match the model.')
        model.compile(optimizer=koptims.RMSprop(lr=TRAIN_LEARNING_RATE),
                      loss=TRAIN_LOSS_FUNCTION,
                      metrics=['accuracy'])

        return model

    def __get_unet_light(self, weights_path: str=None, plot_scheme: bool=False):
        """
        Creates a model with light version of U-Net architecture (less filters).
        :param weights_path: path to the model weights file.
        :param plot_scheme: whether to create the plot of the model architecture.
        :return: Keras model.
        """
        inputs = klayers.Input(self.__input_shape)
        conv1 = klayers.Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = klayers.ReLU()(conv1)
        conv1 = klayers.Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = klayers.ReLU()(conv1)
        pool1 = klayers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = klayers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = klayers.ReLU()(conv2)
        conv2 = klayers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = klayers.ReLU()(conv2)
        pool2 = klayers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = klayers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = klayers.ReLU()(conv3)
        conv3 = klayers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = klayers.ReLU()(conv3)
        pool3 = klayers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = klayers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = klayers.ReLU()(conv4)
        conv4 = klayers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = klayers.ReLU()(conv4)
        drop4 = klayers.Dropout(0.5)(conv4)
        pool4 = klayers.MaxPooling2D(pool_size=(2, 2))(drop4)

        # Bottleneck
        conv5 = klayers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = klayers.ReLU()(conv5)
        conv5 = klayers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = klayers.ReLU()(conv5)
        drop5 = klayers.Dropout(0.5)(conv5)

        up6 = klayers.UpSampling2D(size=(2, 2))(drop5)
        up6 = klayers.Conv2D(256, 2, padding='same', kernel_initializer='he_normal')(up6)
        up6 = klayers.PReLU()(up6)
        merge6 = klayers.concatenate([drop4, up6], axis=3)
        conv6 = klayers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = klayers.ReLU()(conv6)
        conv6 = klayers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = klayers.ReLU()(conv6)

        up7 = klayers.UpSampling2D(size=(2, 2))(conv6)
        up7 = klayers.Conv2D(128, 2, padding='same', kernel_initializer='he_normal')(up7)
        up7 = klayers.ReLU()(up7)
        merge7 = klayers.concatenate([conv3, up7], axis=3)
        conv7 = klayers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = klayers.ReLU()(conv7)
        conv7 = klayers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = klayers.ReLU()(conv7)

        up8 = klayers.UpSampling2D(size=(2, 2))(conv7)
        up8 = klayers.Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(up8)
        up8 = klayers.ReLU()(up8)
        merge8 = klayers.concatenate([conv2, up8], axis=3)
        conv8 = klayers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = klayers.ReLU()(conv8)
        conv8 = klayers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = klayers.ReLU()(conv8)

        up9 = klayers.UpSampling2D(size=(2, 2))(conv8)
        up9 = klayers.Conv2D(32, 2, padding='same', kernel_initializer='he_normal')(up9)
        up9 = klayers.ReLU()(up9)
        merge9 = klayers.concatenate([conv1, up9], axis=3)
        conv9 = klayers.Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = klayers.ReLU()(conv9)
        conv9 = klayers.Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = klayers.ReLU()(conv9)
        conv9 = klayers.Conv2D(2, 3, padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = klayers.ReLU()(conv9)

        outputs = klayers.Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs, outputs, name='u-net')
        if plot_scheme:
            plot_model(model, to_file='U-net_light.png', show_shapes=True)

        if weights_path is not None:
            try:
                model.load_weights(os.path.abspath(weights_path))
            except ValueError:
                raise ValueError('U-net weights does not match the model.')
        model.compile(optimizer=koptims.RMSprop(lr=TRAIN_LEARNING_RATE),
                      loss=TRAIN_LOSS_FUNCTION,
                      metrics=['accuracy'])

        return model

    def __get_unet_author(self,
                          weights_path: str = None,
                          plot_scheme: bool = False,
                          class_weights: dict = None):
        """
        Creates a model with modified version of U-Net architecture (less filters).
        :param weights_path: path to the model weights file.
        :param plot_scheme: whether to create the plot of the model architecture.
        :param class_weights: dict containing class weights.
        :return: Keras model.
        """
        inputs = klayers.Input(self.__input_shape)
        conv1 = klayers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = klayers.BatchNormalization()(conv1)
        conv1 = klayers.LeakyReLU(0.3)(conv1)
        conv1 = klayers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = klayers.BatchNormalization()(conv1)
        conv1 = klayers.LeakyReLU(0.3)(conv1)
        pool1 = klayers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = klayers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = klayers.BatchNormalization()(conv2)
        conv2 = klayers.LeakyReLU(0.3)(conv2)
        conv2 = klayers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = klayers.BatchNormalization()(conv2)
        conv2 = klayers.LeakyReLU(0.3)(conv2)
        pool2 = klayers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = klayers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = klayers.BatchNormalization()(conv3)
        conv3 = klayers.LeakyReLU(0.3)(conv3)
        conv3 = klayers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = klayers.BatchNormalization()(conv3)
        conv3 = klayers.LeakyReLU(0.3)(conv3)
        pool3 = klayers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = klayers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = klayers.BatchNormalization()(conv4)
        conv4 = klayers.LeakyReLU(0.3)(conv4)
        conv4 = klayers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = klayers.BatchNormalization()(conv4)
        conv4 = klayers.LeakyReLU(0.3)(conv4)
        pool4 = klayers.MaxPooling2D(pool_size=(2, 2))(conv4)

        # Bottleneck
        conv5 = klayers.Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = klayers.BatchNormalization()(conv5)
        conv5 = klayers.LeakyReLU(0.3)(conv5)
        conv5 = klayers.Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = klayers.BatchNormalization()(conv5)
        conv5 = klayers.LeakyReLU(0.3)(conv5)

        up6 = klayers.UpSampling2D(size=(2, 2))(conv5)
        up6 = klayers.Conv2D(512, 2, padding='same', kernel_initializer='he_normal')(up6)
        up6 = klayers.BatchNormalization()(up6)
        up6 = klayers.LeakyReLU(0.3)(up6)
        merge6 = klayers.concatenate([conv4, up6], axis=3)
        conv6 = klayers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = klayers.BatchNormalization()(conv6)
        conv6 = klayers.LeakyReLU(0.3)(conv6)
        conv6 = klayers.Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = klayers.BatchNormalization()(conv6)
        conv6 = klayers.LeakyReLU(0.3)(conv6)

        up7 = klayers.UpSampling2D(size=(2, 2))(conv6)
        up7 = klayers.Conv2D(256, 2, padding='same', kernel_initializer='he_normal')(up7)
        up7 = klayers.BatchNormalization()(up7)
        up7 = klayers.LeakyReLU(0.3)(up7)
        merge7 = klayers.concatenate([conv3, up7], axis=3)
        conv7 = klayers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = klayers.BatchNormalization()(conv7)
        conv7 = klayers.LeakyReLU(0.3)(conv7)
        conv7 = klayers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = klayers.BatchNormalization()(conv7)
        conv7 = klayers.LeakyReLU(0.3)(conv7)

        up8 = klayers.UpSampling2D(size=(2, 2))(conv7)
        up8 = klayers.Conv2D(128, 2, padding='same', kernel_initializer='he_normal')(up8)
        up8 = klayers.BatchNormalization()(up8)
        up8 = klayers.LeakyReLU(0.3)(up8)
        merge8 = klayers.concatenate([conv2, up8], axis=3)
        conv8 = klayers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = klayers.BatchNormalization()(conv8)
        conv8 = klayers.LeakyReLU(0.3)(conv8)
        conv8 = klayers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = klayers.BatchNormalization()(conv8)
        conv8 = klayers.LeakyReLU(0.3)(conv8)

        up9 = klayers.UpSampling2D(size=(2, 2))(conv8)
        up9 = klayers.Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(up9)
        up9 = klayers.BatchNormalization()(up9)
        up9 = klayers.LeakyReLU(0.3)(up9)
        merge9 = klayers.concatenate([conv1, up9], axis=3)
        conv9 = klayers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = klayers.BatchNormalization()(conv9)
        conv9 = klayers.LeakyReLU(0.3)(conv9)
        conv9 = klayers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = klayers.BatchNormalization()(conv9)
        conv9 = klayers.LeakyReLU(0.3)(conv9)
        conv9 = klayers.Conv2D(2, 3, padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = klayers.BatchNormalization()(conv9)
        conv9 = klayers.LeakyReLU(0.3)(conv9)

        outputs = klayers.Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs, outputs, name='u-net')
        if plot_scheme:
            plot_model(model, to_file='U-net_modified.png', show_shapes=True)

        if weights_path is not None:
            try:
                model.load_weights(os.path.abspath(weights_path))
            except ValueError:
                raise ValueError('U-net weights does not match the model.')

        if class_weights is not None:
            loss_function = mask_weighted_loss(TRAIN_LOSS_FUNCTION, class_weights)
        else:
            loss_function = TRAIN_LOSS_FUNCTION
        model.compile(optimizer=koptims.RMSprop(lr=TRAIN_LEARNING_RATE),
                      loss=loss_function,
                      metrics=['accuracy'])

        return model

    def __get_fgsegnet(self):
        """
        Creates a model with FgSegNet architecture.
        :return:
        """
        pass

    def get_model(self, model_type: str,
                  weights_path: str = None,
                  plot_scheme: bool = False,
                  class_weights: dict = None):
        """
        Creates a Keras model for lungs foreground segmentation task.
        :param model_type: one of the predefined model types.
        :param weights_path: path to the model weights file.
        :param plot_scheme: whether to create the plot of the model architecture.
        :param class_weights: dict containing class weights.
        :return: an instance of Keras Model class.
        """
        # TODO: KostinKA - if weights are not None call Keras load_model function
        if model_type == ModelType.U_NET.value:
            model = self.__get_unet_author(weights_path=weights_path, plot_scheme=plot_scheme,
                                           class_weights=class_weights)
            # model = self.__get_unet_light(weights_path=weights_path, plot_scheme=plot_scheme)
            # model = self.__get_unet(weights_path=weights_path, plot_scheme=plot_scheme)
        elif model_type == ModelType.FG_SEG_NET.value:
            raise NotImplementedError()
            # model = self.__get_fgsegnet()
        else:
            raise ValueError('Unable to create a model of type \'{}\''.format(model_type))

        return model


class FilesManager:
    def __init__(self, val_size: float, file_extensions: list, shuffle: bool=False, random_seed: int=None):
        self.__val_size = val_size
        self.__file_extensions = file_extensions
        self.__shuffle = shuffle
        self.__random_seed = random_seed

    def __extract_file_paths(self, folder_path: str) -> np.ndarray:
        """
        Extracts paths to files of specified extensions from the folder.
        :param folder_path: path to the folder.
        :return: numpy array of extracted file paths.
        """
        source_folder = os.path.abspath(folder_path)

        file_names = sorted(os.listdir(source_folder))
        file_paths = []
        for fn in file_names:
            if os.path.splitext(fn)[1] not in self.__file_extensions:
                continue

            file_path = os.path.join(source_folder, fn)
            file_paths.append(file_path)

        if len(file_paths) == 0:
            raise ValueError('Target folder {folder} does not contain any files meeting'
                             ' the required extensions: {extensions}.'
                             .format(folder=source_folder, extensions=self.__file_extensions))

        return np.array(file_paths)

    def __split_paths(self, primary_paths: np.ndarray, secondary_paths: np.ndarray=None) \
            -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Performs split input paths into train and validation subsets.
        :param primary_paths: numpy array of file paths - X.
        :param secondary_paths: numpy array of file paths - y.
        :return: tuple of numpy arrays: (train primary paths, val primary paths,
         train secondary paths of None, val secondary paths or None).
        """
        if secondary_paths is None:
            train_paths, val_paths = train_test_split(
                primary_paths,
                test_size=self.__val_size,
                shuffle=self.__shuffle,
                random_state=self.__random_seed
            )
            return train_paths, val_paths, None, None
        else:
            train_primary, val_primary, train_secondary, val_secondary = train_test_split(
                primary_paths,
                secondary_paths,
                test_size=self.__val_size,
                shuffle=self.__shuffle,
                random_state=self.__random_seed
            )
            return train_primary, val_primary, train_secondary, val_secondary

    def prepare_divided_files_paths(self, primary_folder: str, secondary_folder: str=None) \
            -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Prepares file paths from folder divided into train and val subsets.
        :param primary_folder: path to the folder containing primary set of files - X.
        :param secondary_folder: path to the folder containing secondary set of files - y.
        :return: tuple of numpy arrays: (train primary paths, val primary paths,
         train secondary paths of None, val secondary paths or None).
        """
        if secondary_folder is None:
            file_paths = self.__extract_file_paths(primary_folder)
            train_paths, val_paths, _, _ = self.__split_paths(file_paths)
            return train_paths, val_paths, None, None
        else:
            primary_paths = self.__extract_file_paths(primary_folder)
            secondary_paths = self.__extract_file_paths(secondary_folder)
            train_primary, val_primary, train_secondary, val_secondary = self.__split_paths(
                primary_paths,
                secondary_paths
            )
            return train_primary, val_primary, train_secondary, val_secondary

    def prepare_files_paths(self, primary_folder: str, secondary_folder: str=None) -> (np.ndarray, np.ndarray):
        """
        Prepares file paths from folders.
        :param primary_folder: path to the folder containing primary set of files - X.
        :param secondary_folder: path to the folder containing secondary set of files - y.
        :return: tuple of numpy arrays: (primary paths, secondary paths or None).
        """
        if secondary_folder is None:
            file_paths = self.__extract_file_paths(primary_folder)
            return file_paths, None
        else:
            primary_paths = self.__extract_file_paths(primary_folder)
            secondary_paths = self.__extract_file_paths(secondary_folder)
            return primary_paths, secondary_paths


class IOHelper:
    def __init__(self):
        pass

    def clear_folder(self, folder_path: str):
        """
        Removes all files from the folder.
        :param folder_path: path to the folder.
        """
        abs_folder_path = os.path.abspath(folder_path)
        if os.path.exists(abs_folder_path):
            entries = sorted(os.listdir(abs_folder_path))

            for entry in entries:
                entry_path = os.path.join(abs_folder_path, entry)
                if os.path.isdir(entry_path):
                    self.clear_folder(entry_path)
                    os.removedirs(entry_path)
                else:
                    os.remove(entry_path)

    @staticmethod
    def create_folder(folder_path: str):
        """
        Creates a folder.
        :param folder_path: path to the folder.
        """
        abs_folder_path = os.path.abspath(folder_path)
        if not os.path.exists(abs_folder_path):
            os.makedirs(abs_folder_path)

    @staticmethod
    def get_file_clear_name(file_path: str) -> str:
        """
        Extracts clear filename without extension.
        :param file_path: path to the file.
        """
        filename = os.path.basename(os.path.abspath(file_path))
        filename_without_ext = os.path.splitext(filename)[0]

        return filename_without_ext

    @staticmethod
    def save_as_npy(folder: str, name: str, array: np.ndarray) -> str:
        """
        Saves a numpy array as a file in NPY format.
        :param folder: path to the folder where is needed to save a file.
        :param name: desired name of the file (with or without extension).
        :param array: numpy array.
        :return: absolute path to the saved file.
        """
        filename_without_ext = os.path.splitext(name)[0]
        saving_filename = filename_without_ext + '.npy'
        saving_path = os.path.join(os.path.abspath(folder), saving_filename)
        np.save(saving_path, array)

        return saving_path

    @staticmethod
    def load_from_npy(file_path: str):
        """
        Loads a numpy array from the file.
        :param file_path: path to a file in NPY format.
        :return: numpy array.
        """
        return np.load(os.path.abspath(file_path))

    @staticmethod
    def get_files_change_time(folder_path: str) -> dict:
        """
        Calculates information about file paths modification time.
        :param folder_path: path to the folder where files are stored.
        :return: dictionary of files modification time: key - modification time, value - file path.
        """
        source_folder = os.path.abspath(folder_path)
        file_names = sorted(os.listdir(source_folder))
        file_paths = [os.path.join(source_folder, fn) for fn in file_names]

        file_changed_dict = {}
        for fp in file_paths:
            file_changed_dict[os.path.getmtime(fp)] = fp

        return file_changed_dict


class DataManager:
    def __init__(self, desired_shape: tuple):
        self.__desired_shape = desired_shape

        self.__io_helper = IOHelper()

    def __load_and_unify_gen(self, image_paths: np.ndarray) -> (np.ndarray, str):
        """
        Generator for loading images and unify them according to desired shape.
        :param image_paths: numpy array of image file paths.
        :return: tuple: (numpy array of a grayscale image (height, width, 1), image file name without extension).
        """
        for img_path in image_paths:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            filename = IOHelper.get_file_clear_name(img_path)
            if not isinstance(img, np.ndarray):
                raise FileNotFoundError('Unable to load image: {}'.format(img_path))

            desired_rows, desired_cols, desired_channels = self.__desired_shape

            # Decrease image shape if needed
            rows_to_decrease = min(desired_rows, img.shape[0])
            cols_to_decrease = min(desired_cols, img.shape[1])
            changed_img = cv2.resize(img, (cols_to_decrease, rows_to_decrease))  # new width and height

            # Increase image shape if needed
            rows_to_increase = max(desired_rows, img.shape[0])
            cols_to_increase = max(desired_cols, img.shape[1])
            rows_diff = abs(img.shape[0] - rows_to_increase)
            cols_diff = abs(img.shape[1] - cols_to_increase)

            if rows_diff % 2 == 0:
                top_rows_pad = int(rows_diff / 2)
                bottom_rows_pad = int(rows_diff / 2)
            else:
                top_rows_pad = int(math.floor(rows_diff / 2))
                bottom_rows_pad = int(math.floor(rows_diff / 2)) + 1
            if cols_diff % 2 == 0:
                left_cols_pad = int(cols_diff / 2)
                right_cols_pad = int(cols_diff / 2)
            else:
                left_cols_pad = int(math.floor(cols_diff / 2))
                right_cols_pad = int(math.floor(cols_diff / 2)) + 1

            if desired_channels == 1:
                pad_color = 0
            else:
                pad_color = np.zeros((3,), dtype=np.uint8)
            changed_img = cv2.copyMakeBorder(
                changed_img,
                top_rows_pad,
                bottom_rows_pad,
                left_cols_pad,
                right_cols_pad,
                cv2.BORDER_CONSTANT,
                pad_color
            )

            yield changed_img, filename

    def prepare_images(self,
                       image_paths: np.ndarray,
                       target_folder: str,
                       normalize: bool = False) -> np.ndarray:
        """
        Unifies images with a predefined shape and save them in the folder as files in NPY format.
        :param image_paths: numpy array of image file paths.
        :param target_folder: path to the folder where the results should be saved.
        :param normalize: whether to normalize images.
        :return: numpy array of paths to saved NPY files
        """
        self.__io_helper.create_folder(target_folder)
        self.__io_helper.clear_folder(target_folder)

        sample_paths = []
        for img_array, filename in self.__load_and_unify_gen(image_paths):
            img_array = self.reshape_array_for_nn(img_array)
            if normalize:
                img_array = self.normalize_image(img_array)

            ready_sample_path = IOHelper.save_as_npy(target_folder, filename, img_array)
            sample_paths.append(ready_sample_path)

        return np.array(sample_paths)

    def prepare_masks(self,
                      mask_paths: np.ndarray,
                      target_folder: str,
                      normalize: bool = False,
                      balanced: bool = False) -> (np.ndarray, dict):
        """
        Unifies image masks with a predefined shape and save them in the folder as files
            in NPY format. In addition, it calculates class weights for class balance.
        :param mask_paths: numpy array of mask file paths.
        :param target_folder: path to the folder where the results should be saved.
        :param normalize: whether to normalize masks.
        :param balanced: whether to calculate class weights to balance classes.
        :return: tuple: (numpy array of paths to saved NPY files, dict of class weights or None).
        """
        self.__io_helper.create_folder(target_folder)
        self.__io_helper.clear_folder(target_folder)

        sample_paths, class_counts = [], []
        for msk_array, filename in self.__load_and_unify_gen(mask_paths):
            msk_array = self.reshape_array_for_nn(msk_array)
            if normalize:
                msk_array = self.normalize_image(msk_array)
                msk_array = self.binarize_image(msk_array, BINARIZATION_THRESHOLD_NORM, 0.0, 1.0)
            else:
                msk_array = self.binarize_image(msk_array, BINARIZATION_THRESHOLD_IMAGE, 0, 255)

            if balanced:
                class_count = self.calculate_class_count(msk_array)
                class_counts.append(class_count)

            ready_sample_path = IOHelper.save_as_npy(target_folder, filename, msk_array)
            sample_paths.append(ready_sample_path)

        if len(class_counts) != 0:
            class_weights = self.calculate_class_weights(class_counts)
        else:
            class_weights = None

        return np.array(sample_paths), class_weights

    @staticmethod
    def reshape_array_for_nn(image: np.ndarray) -> np.ndarray:
        """
        Reshapes an array according to neural net requirements - add third axis.
        :param image: numpy array of shape (height, width).
        :return: numpy array of shape (height, width, 1).
        """
        if len(image.shape) == 3:
            return image.copy()
        elif len(image.shape) == 2:
            return np.expand_dims(image, axis=2)
        else:
            raise ValueError('Unable to reshape image with shape {}'.format(image.shape))

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """
        Performs an image normalization in [0, 1] range.
        :param image: grayscale image numpy array of shape (height, width, 1).
        :return: grayscale image numpy array of shape (height, width, 1).
        """
        return (image / 255).astype(np.float32)

    @staticmethod
    def denormalize_image(image: np.ndarray) -> np.ndarray:
        """
        Performs an image normalization from [0, 1] range to [0, 255] range.
        :param image: grayscale image numpy array of shape (height, width, 1).
        :return: grayscale image numpy array of shape (height, width, 1).
        """
        return (image * 255).astype(np.uint8)

    @staticmethod
    def binarize_image(image: np.ndarray, threshold, min_value, max_value) -> np.ndarray:
        """
        Performs image binarization according to parameters.
        :param image: grayscale image numpy array of shape (height, width, 1).
        :param threshold: threshold value for binarization.
        :param min_value: minimum value of the result binarized image.
        :param max_value: maximum value of the result binarized image.
        :return: grayscale image numpy array of shape (height, weight, 1).
        """
        bin_image = image.copy()
        bin_image[bin_image > threshold] = max_value
        bin_image[bin_image <= threshold] = min_value

        return bin_image

    @staticmethod
    def calculate_class_count(image: np.ndarray) -> dict:
        """
        Creates a dict containing count of image array values:
         key: unique image array value, value: count of such values in the image.
        :param image: grayscale image numpy array of shape (height, width, 1).
        :return: dict of image value counts.
        """
        labels, count = np.unique(image.ravel(), return_counts=True)
        class_count = {l: c for l, c in zip(labels, count)}

        return class_count

    @staticmethod
    def calculate_class_weights(class_counts: list) -> dict:
        """
        Creates a dict containing information about class weights to balance classes:
         key: index of class value, value: >= 1.0 class weight.
        :param class_counts: list of dicts containing image value counts.
        :return: class weights dictionary.
        """
        # Calculate the total number of samples for each class
        total_class_count = {}
        max_count = -sys.maxsize - 1
        for cc in class_counts:
            for key in cc.keys():
                if key in total_class_count.keys():
                    total_class_count[key] = total_class_count[key] + cc[key]
                else:
                    total_class_count[key] = cc[key]

                if total_class_count[key] > max_count:
                    max_count = total_class_count[key]

        # Create class weights dict: class_idx: weight
        class_weights = {}
        for i, key in enumerate(sorted(total_class_count.keys())):
            class_weights[i] = float(max_count / total_class_count[key])

        return class_weights


class FilesCleanerCallback(kcallbacks.Callback):
    """
    Keras Callback for cleaning special folder
     - keep only specified number of last created files.
    """

    def __init__(self, working_folder: str, files_limit: int, file_extensions: list):
        """
        :param working_folder: path to the folder where files are storing.
        :param files_limit: number of files to keep.
        :param file_extensions: list of required file extensions including a point, e.g. '.h5'.
        """
        super().__init__()

        self.__io_helper = IOHelper()

        self.__working_folder = os.path.abspath(working_folder)
        self.__allowed_extensions = file_extensions

        self.__files_limit = int(files_limit)
        if self.__files_limit <= 0:
            raise ValueError('Number of files to keep should be more than 0.')

    def on_train_begin(self, logs=None):
        self.__io_helper.clear_folder(self.__working_folder)

    def on_epoch_end(self, epoch, logs=None):
        file_changes = self.__io_helper.get_files_change_time(self.__working_folder)

        file_changed_dict = {}
        for k, v in file_changes.items():
            if os.path.splitext(os.path.basename(v))[1] in self.__allowed_extensions:
                file_changed_dict[k] = v

        files_num = len(file_changed_dict.keys())
        if files_num > self.__files_limit:
            old_file_keys = sorted(file_changed_dict.keys())[:(files_num - self.__files_limit)]
            for key in old_file_keys:
                os.remove(file_changed_dict[key])


class InterResultsCallback(kcallbacks.Callback):
    def __init__(self, working_folder: str, val_paths: np.ndarray, val_samples_num: int):
        """
        :param working_folder: path to the folder where results should be created.
        :param val_paths: numpy array of paths to validation samples.
        :param val_samples_num: desired number of validation samples to get.
        """
        super().__init__()

        self.__working_folder = working_folder
        self.__val_paths = val_paths
        self.__samples_num = int(val_samples_num)
        self.__batch_size = 1

        self.__io_helper = IOHelper()
        self.__sample_names = None
        self.__samples = None

    def on_train_begin(self, logs=None):
        self.__io_helper.clear_folder(self.__working_folder)
        indices = np.arange(self.__val_paths.shape[0])
        np.random.shuffle(indices)
        indices = indices[:self.__samples_num]

        sample_paths = list(self.__val_paths[indices])
        self.__sample_names = [os.path.splitext(os.path.basename(sp))[0] for sp in sample_paths]
        self.__samples = np.array([self.__io_helper.load_from_npy(sp) for sp in sample_paths])

    def on_epoch_end(self, epoch, logs=None):
        inter_results_folder = os.path.join(self.__working_folder, 'epoch-{:04d}'.format(epoch))
        self.__io_helper.create_folder(inter_results_folder)

        predictions = self.model.predict(self.__samples, batch_size=self.__batch_size)
        for pred, name in zip(predictions, self.__sample_names):
            pred_mask = np.reshape(pred, (pred.shape[0], pred.shape[1]))

            pred_mask = DataManager.denormalize_image(pred_mask)
            pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(inter_results_folder, name + '.png'), pred_mask)


class TrainingManager:
    def __init__(self, batch_size: int, epochs: int, identifier: str='model'):
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__identifier = identifier

        checkpoints_name = identifier + '_checkpoints'
        logs_name = 'logs'
        inter_results_name = identifier + '_inters'
        current_folder = os.path.abspath(os.path.dirname(__file__))

        self.__checkpoint_pattern = identifier + '_weights_{epoch:04d}_{val_loss:.4f}.h5'
        self.__keep_checkpoints = 3
        self.__checkpoint_extensions = WEIGHTS_EXTENSIONS
        self.__log_pattern = identifier + '_{datetime}.log'
        self.__inter_results_num = TRAIN_INTER_RESULTS_NUM

        self.__checkpoints_folder = os.path.join(current_folder, checkpoints_name)
        self.__logs_folder = os.path.join(current_folder, logs_name)
        self.__inters_folder = os.path.join(current_folder, inter_results_name)

        self.__shuffle = TRAIN_SHUFFLE
        self.__verbosity = TRAIN_VERBOSITY

    def __create_callbacks(self, checkpoints: bool=True, checkpoints_cleaning: bool=True, log: bool=True,
                           inter_results: bool=False, val_paths: np.ndarray=None) -> list:
        """
        Creates a list of Keras callbacks for training.
        :param checkpoints: whether to create a callback for checkpoints saving.
        :param checkpoints_cleaning: whether to create a callback for
         keeping only defined number of last model checkpoints.
        :param log: whether to create a callback for writing training logs.
        :param inter_results: whether to create InterResultsCallback.
        :param val_paths: numpy array of paths to validation samples,
         it make sense when inter_results is True.
        :return: list of Keras callbacks.
        """
        callbacks = []

        if checkpoints:
            os.makedirs(self.__checkpoints_folder, exist_ok=True)

            checkpointer = kcallbacks.ModelCheckpoint(
                os.path.join(self.__checkpoints_folder, self.__checkpoint_pattern),
                monitor='val_loss',
                verbose= self.__verbosity,
                save_best_only=True,
                save_weights_only=True,
                mode='auto',
                period=1)
            callbacks.append(checkpointer)

            if checkpoints_cleaning:
                checkpoint_cleaner = FilesCleanerCallback(self.__checkpoints_folder,
                                                          self.__keep_checkpoints,
                                                          self.__checkpoint_extensions)
                callbacks.append(checkpoint_cleaner)

        if log:
            os.makedirs(self.__logs_folder, exist_ok=True)
            logger = kcallbacks.CSVLogger(
                os.path.join(
                	self.__logs_folder,
                	self.__log_pattern.format(datetime=datetime.now().strftime('%Y-%m-%d %H_%M_%S'))),
                separator=',',
                append=False)
            callbacks.append(logger)

        if inter_results:
            os.makedirs(self.__inters_folder, exist_ok=True)

            inter_results_callback = InterResultsCallback(self.__inters_folder,
                                                          val_paths,
                                                          self.__inter_results_num)
            callbacks.append(inter_results_callback)

        return callbacks

    def __generate_samples(self, img_paths: np.ndarray, mask_paths: np.ndarray, shuffle: bool=False) \
            -> (np.ndarray, np.ndarray):
        """
        Generator to create batches of X, Y samples stored in NPY files.
        :param img_paths: numpy array of paths to image samples stored as NPY files.
        :param mask_paths: numpy array of paths to mask samples stored as NPY files.
        :param shuffle: whether to shuffle paths before running the generation.
        """
        if img_paths.shape[0] != mask_paths.shape[0]:
            raise ValueError('Unable to generate samples: number of image paths'
                             ' ({imgs_number}) != number of mask paths ({msks_number})'
                             .format(imgs_number=img_paths.shape[0], msks_number=mask_paths.shape[0]))

        while True:
            indices = np.arange(img_paths.shape[0])
            if shuffle:
                np.random.shuffle(indices)

            batch_count = int(math.ceil(img_paths.shape[0] / self.__batch_size))
            for i in range(batch_count):
                indices_to_take = indices[(i * self.__batch_size): ((i * self.__batch_size) + self.__batch_size)]

                X_paths = img_paths[indices_to_take]
                y_paths = mask_paths[indices_to_take]

                X_samples = np.array([IOHelper.load_from_npy(p) for p in X_paths])
                y_samples = np.array([IOHelper.load_from_npy(p) for p in y_paths])

                yield X_samples, y_samples

    def train(self, model: Model,
              X_train: np.ndarray,
              X_val: np.ndarray,
              y_train: np.ndarray,
              y_val: np.ndarray) -> str:
        """
        Runs training a Keras model on data included images and masks.
        :param model: an instance of the Keras Model.
        :param X_train: numpy array of paths to training image samples stored as NPY files.
        :param X_val: numpy array of paths to validation image samples stored as NPY files.
        :param y_train: numpy array of paths to training mask samples stored as NPY files.
        :param y_val: numpy array of paths to validation mask samples stored as NPY files.
        :return: path to the best model weights file.
        """
        model.fit_generator(self.__generate_samples(X_train, y_train, self.__shuffle),
                            steps_per_epoch=int(math.ceil(X_train.shape[0] / self.__batch_size)),
                            epochs=self.__epochs,
                            verbose=self.__verbosity,
                            callbacks=self.__create_callbacks(checkpoints=True,
                                                              checkpoints_cleaning=True,
                                                              log=True,
                                                              inter_results=TRAIN_INTER_RESULTS,
                                                              val_paths=X_val),
                            validation_data=self.__generate_samples(X_val, y_val, self.__shuffle),
                            validation_steps=int(math.ceil(X_val.shape[0] / self.__batch_size)))

        # Find the best model weights checkpoint
        file_changes = IOHelper.get_files_change_time(self.__checkpoints_folder)
        file_changed_dict = {}
        for k, v in file_changes.items():
            if os.path.splitext(os.path.basename(v))[1] in self.__checkpoint_extensions:
                file_changed_dict[k] = v

        oldest_file_path = file_changed_dict[sorted(file_changed_dict.keys())[-1]]
        return oldest_file_path


class PredictionManager:
    def __init__(self):
        self.__io_helper = IOHelper()

        self.__batch_size = 1  # Working only with batch_size == 1
        self.__verbosity = TEST_VERBOSITY

    def __generate_samples(self, img_paths: np.ndarray) -> np.ndarray:
        """
        Generator to create batches of image samples for model prediction.
        :param img_paths: numpy array of paths to image samples stored as NPY files.
        :return: numpy array of images - batch of image numpy arrays.
        """
        for path in img_paths:
            sample = self.__io_helper.load_from_npy(path)
            sample = np.expand_dims(sample, axis=0)
            yield sample

    def __generate_targets(self, msk_paths: np.ndarray) -> np.ndarray:
        """
        Generator to load targets from NPY files.
        :param msk_paths: numpy array of paths to target samples stored as NPY files.
        :return: target numpy array.
        """
        for path in msk_paths:
            target = self.__io_helper.load_from_npy(path)
            yield target

    def generate_outputs(self, model: Model, X_test: np.ndarray, y_test: np.ndarray) \
            -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Runs prediction of the Keras model on images data.
        :param model: an instance of the Keras Model.
        :param X_test: numpy array of paths to image samples stored as NPY files.
        :param y_test: numpy array of paths to mask samples stored as NPY files.
        :return: tuple: (input sample numpy array, numpy array of true output mask,
         numpy array of predicted output mask).
        """
        for sample, true, pred in zip(
                self.__generate_samples(X_test),
                self.__generate_targets(y_test),
                model.predict_generator(self.__generate_samples(X_test),
                                        steps=int(math.ceil(X_test.shape[0] / self.__batch_size)),
                                        verbose=self.__verbosity)
        ):
            yield np.reshape(sample, (sample.shape[1], sample.shape[2], sample.shape[3])), true, pred


def train_network(images_folder: str, masks_folder: str, model_type: str,
                  weights_path: str=None, random_seed: int=None) -> str:
    """
    Performs training of a segmentation neural net model.
    :param images_folder: path to the folder where images are stored.
    :param masks_folder: path to the folder where masks are stored.
    :param model_type: predefined type of the model to create.
    :param weights_path: path to the model weights file.
    :param random_seed: random seed for scikit-learn, numpy and tensorflow.
    :return: path to the trained model weights file.
    """
    # Prepare file paths
    files_manager = FilesManager(FILES_VAL_SIZE, IMG_EXTENSIONS, shuffle=FILES_SHUFFLE, random_seed=random_seed)
    train_images, val_images, train_masks, val_masks = files_manager.prepare_divided_files_paths(images_folder,
                                                                                                 masks_folder)

    # Prepare paths to ready binary data (in NPY files)
    data_manager = DataManager(IMG_SHAPE)
    X_train = data_manager.prepare_images(train_images, IMG_TRAIN_FOLDER,
                                          normalize=IMG_NORMALIZATION)
    X_val = data_manager.prepare_images(val_images, IMG_VAL_FOLDER,
                                        normalize=IMG_NORMALIZATION)
    y_train, class_weights = data_manager.prepare_masks(train_masks, MASK_TRAIN_FOLDER,
                                                        normalize=MASK_NORMALIZATION,
                                                        balanced=MASK_BALANCING)
    y_val, _ = data_manager.prepare_masks(val_masks, MASK_VAL_FOLDER,
                                          normalize=MASK_NORMALIZATION)

    # Prepare a model
    model_manager = ModelManager(IMG_SHAPE)
    model = model_manager.get_model(model_type, weights_path=weights_path,
                                    plot_scheme=MODEL_PLOT_SCHEME, class_weights=class_weights)

    # Perform training the model on the prepared data
    train_manager = TrainingManager(TRAIN_BATCH_SIZE, TRAIN_EPOCHS, identifier=model_type)
    weights_file_path = train_manager.train(model, X_train, X_val, y_train, y_val)

    return weights_file_path


def test_network(images_folder: str, masks_folder: str, model_type: str,
                 weights_path: str=None, random_seed: int=None) -> str:
    """
    Performs testing of a segmentation neural net model:
     creates an input image with true and predicted regions.
    :param images_folder: path to the folder containing input images.
    :param masks_folder: path to the folder containing true region masks for input images.
    :param model_type: predefined type of the model to create.
    :param weights_path: path to the model weights file.
    :param random_seed: random seed for scikit-learn, numpy and tensorflow.
    :return: path to the folder where the results are saved.
    """
    # Prepare output folder
    current_folder = os.path.abspath(os.path.dirname(__file__))
    output_folder = os.path.join(current_folder, TEST_OUTPUT_FOLDER)
    io_helper = IOHelper()
    io_helper.create_folder(output_folder)
    io_helper.clear_folder(output_folder)

    # Prepare file paths
    files_manager = FilesManager(FILES_VAL_SIZE, IMG_EXTENSIONS, shuffle=False, random_seed=random_seed)
    image_paths, mask_paths = files_manager.prepare_files_paths(images_folder, masks_folder)

    # Prepare paths to ready binary data (in NPY files)
    data_manager = DataManager(IMG_SHAPE)
    X_test = data_manager.prepare_images(image_paths, IMG_TEST_FOLDER,
                                         normalize=IMG_NORMALIZATION)
    y_test, _ = data_manager.prepare_masks(mask_paths, MASK_TEST_FOLDER,
                                           normalize=MASK_NORMALIZATION)

    # Prepare a model
    model_manager = ModelManager(IMG_SHAPE)
    model = model_manager.get_model(model_type, weights_path=weights_path, plot_scheme=MODEL_PLOT_SCHEME)

    # Perform prediction of the masks
    conf_matrix = np.zeros((2, 2), dtype=np.int64)
    prediction_manager = PredictionManager()
    for i, (sample, true, pred) in enumerate(prediction_manager.generate_outputs(model, X_test, y_test)):
        # Reshaping - remove third axis
        X = np.reshape(sample, (sample.shape[0], sample.shape[1]))
        y = np.reshape(true, (true.shape[0], true.shape[1]))
        y_pred = np.reshape(pred, (pred.shape[0], pred.shape[1]))

        # Thresholding for masks binarization
        y_pred[y_pred >= TEST_THRESHOLD] = 1.0
        y_pred[y_pred < TEST_THRESHOLD] = 0.0
        y[y >= TEST_THRESHOLD] = 1.0
        y[y < TEST_THRESHOLD] = 0.0

        # Collecting confusion matrix
        conf_matrix += confusion_matrix(y.ravel(), y_pred.ravel()).astype(np.int64)

        # Transform float numpy arrays into images
        X = DataManager.denormalize_image(X)
        y = DataManager.denormalize_image(y)
        y_pred = DataManager.denormalize_image(y_pred)

        # Draw region contours on a sample image
        result_img = cv2.cvtColor(X, cv2.COLOR_GRAY2BGR)
        true_contours, _ = cv2.findContours(y, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        pred_contours, _ = cv2.findContours(y_pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(result_img, true_contours, -1, TEST_TRUE_REGION_BGR_COLOR, 1)
        cv2.drawContours(result_img, pred_contours, -1, TEST_PRED_REGION_BGR_COLOR, 1)

        # Save a result image
        img_name = '{idx}.png'.format(idx=i)
        img_path = os.path.join(output_folder, img_name)
        cv2.imwrite(img_path, result_img)

        # Save true and pred masks
        pred_mask_name = 'pred_{idx}.png'.format(idx=i)
        pred_mask_path = os.path.join(output_folder, pred_mask_name)
        true_mask_name = 'true_{idx}.png'.format(idx=i)
        true_mask_path = os.path.join(output_folder, true_mask_name)
        cv2.imwrite(pred_mask_path, y_pred)
        cv2.imwrite(true_mask_path, y)

    # Calculating metrics by confusion matrix
    tp, fp, fn = conf_matrix[1, 1], conf_matrix[0, 1], conf_matrix[1, 0]
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)

    print('Precision = {:.2f}'.format(precision))
    print('Recall = {:.2f}'.format(recall))
    print('F1 = {:.2f}'.format(f1))

    return output_folder


def main(mode: str, images_folder: str, masks_folder: str, model_type: str,
         weights_path: str=None, random_seed: int=None):
    """
    :param mode: predefined script launch mode.
    :param images_folder: path to the folder where images are stored.
    :param masks_folder: path to the folder where masks are stored.
    :param model_type: predefined type of the model to create.
    :param weights_path: path to the model weights file.
    :param random_seed: random seed for scikit-learn, numpy and tensorflow.
    :return:
    """
    if random_seed is not None:
        rnd_seed = random_seed

        random.seed(rnd_seed)
        np.random.seed(rnd_seed)
        set_random_seed(rnd_seed)
    else:
        rnd_seed = random_seed

    if mode is None:
        raise ValueError('Script launch mode should be set.')
    s_modes = [m.value for m in ScriptMode]
    if mode not in s_modes:
        raise ValueError('Invalid script launch mode: {}'.format(mode))

    if images_folder is None:
        raise ValueError('Path to the images folder should be set.')
    img_folder = os.path.abspath(images_folder)

    if masks_folder is None:
        raise ValueError('Path to the masks folder should be set.')
    msk_folder = os.path.join(masks_folder)

    if model_type is None:
        raise ValueError('Model type should be set.')
    else:
        m_types = [t.value for t in ModelType]
        if model_type not in m_types:
            raise ValueError('Invalid model type: {}'.format(model_type))

    if mode == ScriptMode.TRAIN.value:
        model_weights_path = train_network(img_folder, msk_folder, model_type,
                                           weights_path=weights_path, random_seed=rnd_seed)
        print('The model has been trained. Model weights path: {}'.format(model_weights_path))
    elif mode == ScriptMode.TEST.value:
        if weights_path is None:
            warnings.warn('You run the script in the {} mode but path to'
                          ' the model weights file did not set.'.format(mode))

        results_folder = test_network(img_folder, msk_folder, model_type,
                                      weights_path=weights_path, random_seed=rnd_seed)
        print('Test has been completed. Results are in the folder: {}'.format(results_folder))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='To wotk with foreground segmentation networks')

    parser.add_argument(
        '-m',
        '--mode',
        metavar='MODE_NAME',
        type=str,
        help='The script launch mode. It can be one of the following: train, test.'
    )

    parser.add_argument(
        '-i',
        '--images_folder',
        metavar='path/to/folder/',
        type=str,
        help='Path to the folder containing images.'
    )

    parser.add_argument(
        '-l',
        '--labels_folder',
        metavar='path/to/folder/',
        type=str,
        help='Path to the folder containing segment masks for images.'
    )

    parser.add_argument(
        '-t',
        '--type',
        metavar='MODEL_TYPE',
        type=str,
        help='Model type - one of the following: unet, fgsegnet.'
    )

    parser.add_argument(
        '-w',
        '--weights',
        metavar='path/to/file',
        type=str,
        help='Path to a trained model weights file in an hdf5 or h5 format.'
             ' Default is None.',
        default=None
    )

    parser.add_argument(
        '-r',
        '--random_seed',
        metavar='INTEGER_NUMBER',
        type=int,
        help='Random seed for scikit-learn, numpy and tensorflow.',
        default=None
    )

    args = parser.parse_args()

    main(args.mode, args.images_folder, args.labels_folder, args.type, args.weights, args.random_seed)

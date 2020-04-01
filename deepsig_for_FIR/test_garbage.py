import argparse
import h5py
import json
import keras
import numpy as np
import pickle as pkl
import time
import os

from argparse import Namespace
from CustomModelCheckpoint import CustomModelCheckpoint
from DataGenerator import DataGenerator
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Flatten, Reshape, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, MaxPooling1D, Conv1D
from keras.models import Model
from keras.optimizers import Adam
from  FIRLayer import FIRLayer
from keras.utils import plot_model
from keras.utils.io_utils import HDF5Matrix
# from numba import njit, prange
from sklearn.model_selection import train_test_split
from tqdm import tqdm


inputs = Input(shape=(1, 1024, 2), name='Input')
# FIRLayer(output_dim=slice_size, filter_dim=fir_size, channels=1, verbose=1, input_shape=(slice_size, 2)))
x = FIRLayer(output_dim=(1, 1024, 2), filter_dim=self.args.fir_size, channels=1, verbose=1, input_shape=(1, 1024, 2))(
            inputs, name = 'FIR_layer')
x = Conv2D(64, kernel_size=1, name='Conv_1', trainable = False)(x)
x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name='MaxPool_1', trainable = False)(x)
x = Flatten(name = 'Flatten')(x)
x = Dense(128, activation='selu', name = 'Dense_1')(x)
x = Dense(128, activation='selu', name = 'Dense_2')(x)
x = Dense(24, activation='softmax', name = 'Softmax')(x)
model1 = Model(inputs=inputs, outputs=x)
model1.summary()

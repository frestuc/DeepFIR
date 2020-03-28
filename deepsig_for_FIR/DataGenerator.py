import concurrent.futures
import h5py
import numpy as np
import keras
import os
#import pathos.pools as pp
import pickle
import threading
import time

from keras.utils.io_utils import HDF5Matrix
import Utils


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras.'

    def __init__(self, indexes, batch_size, data_path, shuffle=False):
        'Initialization'
        self.indexes = indexes
        self.batch_size = batch_size
        self.data_path = data_path
        self.shuffle = shuffle
        self.cache = {}
        self.X = HDF5Matrix(self.data_path, 'X')
        self.Y = HDF5Matrix(self.data_path, 'Y')

    def __len__(self):
        'Denotes the number of batches per epoch.'
        return int(np.floor(len(self.indexes) / self.batch_size))


    def __add_to_cache(self, indexes):
        '''Add indexes to cache.'''
        for index in indexes:
            if index not in self.cache:
                self.cache[index] = {'X': self.X[index], 'Y': self.Y[index]}


    def __getitem__(self, index):
        'Generate one batch of data.'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X = np.empty((self.batch_size, 1024, 2))
        y = np.empty((self.batch_size, 24), dtype=int)
        for i, idx in enumerate(indexes):
            X[i,] = self.X[idx]
            y[i] = self.Y[idx]
        X = np.expand_dims(X, 1)
        #print('SHAPE: ', X.shape) 
        #X = np.transpose(X, (0, 1, 3, 2))
        #print(np.argmax(y))
        return X, y

    def __fetch_index(self, index):
        self.x_out = self.X[index]
        self.y_out = self.Y[index]

    #def on_epoch_end(self):
    #    'Updates indexes after each epoch'
    #    self.indexes = self.indexes

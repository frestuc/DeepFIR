import concurrent.futures
import h5py
import numpy as np
import DataGenerator
import keras
import os
#import pathos.pools as pp
import pickle
import threading
import time

from keras.utils.io_utils import HDF5Matrix
import Utils


class DataGeneratorFIR(keras.utils.Sequence):
    'Generates data for Keras.'

    def __init__(self, indexes, batch_size, data_path, shuffle=False, is_2d = False,
                 models_path = None , model_name = 'FIR_model', taps_name = 'phi',
                 FIR_layer_name = 'FIR_layer', num_classes = 24):
        'Initialization'
        self.indexes = indexes
        self.batch_size = batch_size
        self.data_path = data_path
        self.shuffle = shuffle
        self.cache = {}
        self.is_2d = is_2d

        # load FIR taps, this is supposed to be saved as a FIR layer with name FIR_layer_name and taps are named as taps_name:0
        for d in range(num_classes):
            f = h5py.File(os.path.join(models_path, model_name + '_' + str(d) + '.hdf5'), 'r')
            if d == 0:
                shape_var = f['model_weights'][FIR_layer_name][FIR_layer_name][taps_name+':0'].shape
                trivial_dimension = np.argwhere(np.array(shape_var) == 1)[0]
                if len(trivial_dimension):  # this is only if we need to remove dummy dimensions
                    print('Dropping one input dimension')
                    self.shape_FIR = np.delete(shape_var,trivial_dimension)
                self.fir_taps = np.zeros(np.append(self.shape_FIR,num_classes))

            # SALVO il secondo FIR layer name cambia ad ogni modello, cercare un modo di prendere solo quello tramite keys.
            if trivial_dimension == 0:
                self.fir_taps[:, :, d] = f['model_weights'][FIR_layer_name][FIR_layer_name][taps_name + ':0'][0, :, :]
            elif trivial_dimension == 1:
                self.fir_taps[:, :, d] = f['model_weights'][FIR_layer_name][FIR_layer_name][taps_name + ':0'][:, 0, :]
            else:
                self.fir_taps[:, :, d] = f['model_weights'][FIR_layer_name][FIR_layer_name][taps_name + ':0'][:, :, 0]


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
            y[i] = self.Y[idx]

            # To Do, here we need to edit the input via FIR convolution
            X[i,] = self.X[idx]

        if self.is_2d: # this is done just to use 2D models and add a new dimension
            X = np.expand_dims(X, 1)


        # print(X[0])
        # print(np.argmax(y))
        return X, y

    def __fetch_index(self, index):
        self.x_out = self.X[index]
        self.y_out = self.Y[index]

    #def on_epoch_end(self):
    #    'Updates indexes after each epoch'
    #    self.indexes = self.indexes

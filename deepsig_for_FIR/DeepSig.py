import argparse
import h5py
import json
import keras
import numpy as np
import pickle
import os
import time

from argparse import Namespace
from CustomModelCheckpoint import CustomModelCheckpoint
from DataGenerator import DataGenerator
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Flatten, Reshape, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, MaxPooling1D, Conv1D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.utils.io_utils import HDF5Matrix
from numba import njit, prange
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class DeepSig(object):
    '''Class to build, train, and test model described in paper
    Over-the-Air Deep Learning Based Radio Signal Classification
    by O'Shea. Data from https://www.deepsig.io/datasets 2018.01A.'''


    def __init__(self):
        '''Initialize class variables.'''
        self.args = self.parse_arguments()
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.id_gpu)
        self.run()


    def build_model(self):
        '''Build model architecture.'''
        print('*************** Building Model ***************')
        inputs = Input(shape=(1, 1024, 2))
        x = Conv2D(64, kernel_size=1)(inputs)
        print(x)
        #x = Lambda(lambda m: K.squeeze(m, 1))(x)
        x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last')(x)
        print(x)
        #x = Lambda(lambda m: K.expand_dims(m, 1))(x)
        for i in range(6):
            print(x)
            x = Conv2D(64, kernel_size=1)(x)
            #x = Lambda(lambda m: K.squeeze(m, 1))(x)
            x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last')(x)
            #x = Lambda(lambda m: K.expand_dims(m, 1))(x)
        x = Flatten()(x)
        x = Dense(128, activation='selu')(x)
        x = Dense(128, activation='selu')(x)
        x = Dense(24, activation='softmax')(x)
        self.model = Model(inputs=inputs, outputs=x)
        self.model.summary()


    def load_data(self):
        '''Load data from path into framework.'''
        print('*************** Loading Data ***************')
        indexes = range(len(HDF5Matrix(self.args.h5_path, 'X')))
        np.random.shuffle(indexes)
        train_idx = int(len(indexes)*.74)
        valid_idx = int(len(indexes)*.80)
        self.train_indexes = indexes[:train_idx]
        self.valid_indexes = indexes[train_idx:valid_idx]
        self.test_indexes = indexes[valid_idx:]

        self.train_generator = DataGenerator(indexes=self.train_indexes,
                                             batch_size=self.args.batch_size,
                                             data_path=self.args.data_path)
        self.valid_generator = DataGenerator(indexes=self.valid_indexes,
                                             batch_size=self.args.batch_size,
                                             data_path=self.args.data_path)
        self.test_generator = DataGenerator(indexes=self.test_indexes,
                                            batch_size=self.args.batch_size,
                                            data_path=self.args.data_path)        
    


    def train(self):
        '''Train model through Keras framework.'''
        print('*************** Training Model ***************')
        optimizer = Adam(lr=0.0001)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])
        call_backs = []
        checkpoint = CustomModelCheckpoint(
                os.path.join(self.args.save_path, "weights.hdf5"),
                monitor='val_acc', verbose=1, save_best_only=True)
        call_backs.append(checkpoint)
        earlystop_callback = EarlyStopping(
                    monitor='val_acc', min_delta=0, patience=self.args.patience,
                    verbose=1, mode='auto')
        call_backs.append(earlystop_callback)

        start_time = time.time()
        self.model.fit_generator(generator=self.train_generator,
                                 epochs=self.args.epochs,
                                 validation_data=self.valid_generator,
                                 shuffle=False,
                                 callbacks=call_backs,
                                 max_queue_size=100)
        train_time = time.time() - start_time
        print('Time to train model %0.3f s' % train_time)
        self.best_model_path = checkpoint.best_path
       

    def test(self):
        '''Test the trained model.
        X = HDF5Matrix(self.args.data_path, 'X')
        Y = HDF5Matrix(self.args.data_path, 'Y')

        self.model.load_weights('/home/bruno/deepsig/weights.hdf5')
        corr = 0
        for idx in tqdm(self.test_indexes):
            idx = int(idx)
            actual_label = np.argmax(Y[idx])
            x = np.expand_dims(X[idx], 0)
            predic_label = np.argmax(self.model.predict(x))
            if predic_label == actual_label:
                corr += 1
        acc = corr / len(self.test_indexes)
        print 'Test accuracy: ', acc
        '''
        #self.model.load_weights('/home/bruno/deepsig3/weights.hdf5')
        optimizer = Adam(lr=0.0001)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])
        score = self.model.evaluate_generator(self.test_generator, verbose=1)
        print(score)

    def run(self):
        '''Run different steps in model pipeline.'''
        self.build_model()
        self.load_data()
        self.train()
        self.test()


    def parse_arguments(self):
        '''Parse input user arguments.'''
        
        parser = argparse.ArgumentParser(description = 'Train and Validation pipeline',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--id_gpu', type=int, default=2,
                            help='GPU to use.')

        parser.add_argument('--save_path', type=str, default='/home/bruno/deepsig3',
                            help='Path to save weights, model architecture, and logs.')

        parser.add_argument('--h5_path', type=str,
                            default='/mnt/WDMyBook/bruno/deepsig/GOLD_XYZ_OSC.0001_1024.hdf5',
                            help='Path to original h5 file.')
        
        #GOLD_XYZ_OSC.0001_1024
        parser.add_argument('--data_path', type=str,
                            default='/mnt/WDMyBook/bruno/deepsig/GOLD_XYZ_OSC.0001_1024.hdf5',
                            help='Path to data.')
    
        parser.add_argument('--patience', type=int, default=3,
                            help='Early stopping patience.')
        
        parser.add_argument('--batch_size', type=int, default=32,
                            help='Batch size for model optimization.')

        parser.add_argument('--epochs', type=int, default=25,
                            help='Number of epochs to train model.')

        return parser.parse_args()
        
        
if __name__ == '__main__':
    DeepSig()


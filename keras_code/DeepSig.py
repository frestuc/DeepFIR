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

class DeepSig(object):
    '''Class to build, train, and test model described in paper
    Over-the-Air Deep Learning Based Radio Signal Classification
    by O'Shea. Data from https://www.deepsig.io/datasets 2018.01A.
    In this dataset we have X,Y,Z
        x : actual [[I1 Q1], [I2 Q2]] format;
        y : labels in [1 0 0 ... 0 ] format
        z : SNR values from -20 to 30
    Each modulation has 106496 samples with several SNR values'''


    def __init__(self):
        '''Initialize class variables.'''
        self.args = self.parse_arguments()
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.id_gpu)

        self.is_2d = True
        self.num_classes = self.args.num_classes
        self.num_examples_per_class = self.args.num_ex_mod

        if self.args.train_cnn or self.args.train_fir or self.args.train_fir_perdev:
            self.run()
        else:
            print('You are not training any model')


    def build_model_baseline(self):
        '''Build model architecture.'''
        print('*************** Building Baseline Model ***************')
        inputs = Input(shape=(1, 1024, 2), name='Input')
        x = Conv2D(64, kernel_size=1, name='Conv_1')(inputs)
        x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name='MaxPool_1')(x)
        x = Conv2D(64, kernel_size=1, name='Conv_2')(x)
        x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name='MaxPool_2')(x)
        x = Conv2D(64, kernel_size=1, name='Conv_3')(x)
        x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name='MaxPool_3')(x)
        x = Conv2D(64, kernel_size=1, name='Conv_4')(x)
        x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name='MaxPool_4')(x)
        x = Conv2D(64, kernel_size=1, name='Conv_5')(x)
        x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name='MaxPool_5')(x)
        x = Conv2D(64, kernel_size=1, name='Conv_6')(x)
        x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name='MaxPool_6')(x)
        x = Conv2D(64, kernel_size=1, name='Conv_7')(x)
        x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name='MaxPool_7')(x)
        x = Flatten(name='Flatten')(x)
        x = Dense(128, activation='selu', name='Dense_1')(x)
        x = Dense(128, activation='selu', name='Dense_2')(x)
        x = Dense(self.num_classes, activation='softmax', name='Softmax')(x)
        self.model = Model(inputs=inputs, outputs=x)
        self.model.summary()

    def build_model_FIR(self):
        '''Build model architecture.'''
        print('*************** Building Baseline Model with FIR ***************')
        #test

        inputs = Input(shape=(1, 1024, 2), name='Input')
        # FIRLayer(output_dim=slice_size, filter_dim=fir_size, channels=1, verbose=1, input_shape=(slice_size, 2)))
        # x = FIRLayer(output_dim=(1, 1024, 2), filter_dim=self.args.fir_size, channels=1, verbose=1, input_shape=(1, 1024, 2), name = 'FIR_layer')(
        #     inputs)
        if self.args.epsilon:
            x = FIRLayer(output_dim=1024, filter_dim=self.args.fir_size, channels=1, verbose=1,
                         input_shape=(1, 1024, 2), epsilon = self.args.epsilon, name = 'FIR_layer')(inputs)
        else:
            x = FIRLayer(output_dim=1024, filter_dim=self.args.fir_size, channels=1, verbose=1,
                         input_shape=(1, 1024, 2), name='FIR_layer')(inputs)
        x = Conv2D(64, kernel_size=1, name='Conv_1', trainable = False)(x)
        x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name='MaxPool_1', trainable = False)(x)
        x = Conv2D(64, kernel_size=1, name='Conv_2', trainable = False)(x)
        x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name='MaxPool_2', trainable = False)(x)
        x = Conv2D(64, kernel_size=1, name='Conv_3', trainable = False)(x)
        x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name='MaxPool_3', trainable = False)(x)
        x = Conv2D(64, kernel_size=1, name='Conv_4', trainable = False)(x)
        x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name='MaxPool_4', trainable = False)(x)
        x = Conv2D(64, kernel_size=1, name='Conv_5', trainable = False)(x)
        x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name='MaxPool_5', trainable = False)(x)
        x = Conv2D(64, kernel_size=1, name='Conv_6', trainable = False)(x)
        x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name='MaxPool_6', trainable = False)(x)
        x = Conv2D(64, kernel_size=1, name='Conv_7', trainable = False)(x)
        x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name='MaxPool_7', trainable = False)(x)
        x = Flatten(name='Flatten', trainable = False)(x)
        x = Dense(128, activation='selu', name='Dense_1', trainable = False)(x)
        x = Dense(128, activation='selu', name='Dense_2', trainable = False)(x)
        x = Dense(self.num_classes, activation='softmax', name='Softmax', trainable = False)(x)
        self.model = Model(inputs=inputs, outputs=x)
        self.model.summary()
        # print(os.path.join(self.args.save_path, self.args.bl_model_name))
        self.model.load_weights(os.path.join(self.args.save_path, self.args.bl_model_name), by_name=True, skip_mismatch=True, reshape=False)
        # exit()


    def load_data(self):
        '''Load data from path into framework.'''
        print('*************** Loading Data ***************')
        print('Load indexes: %r' % self.args.load_indexes)
        if self.args.load_indexes:
            print('--------- Loading from File indexes.pkl ---------')
            # Getting back the objects:
            with open('indexes.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
                print(f)
                self.train_indexes_BL, \
                self.train_indexes_FIR, \
                self.valid_indexes_BL, \
                self.valid_indexes_FIR, \
                self.test_indexes = pkl.load(f)
        else:
            if ~os.path.exists('indexes.pkl'):
                print('--------- Creating indexes and saving them in indexes.pkl -----------')
                indexes_start = np.array(range(self.num_examples_per_class)) # this goes from 0 to 106495
                np.random.shuffle(indexes_start)
                indexes = indexes_start

                train_idx_baseline = int(len(indexes) * .54)
                valid_idx_baseline = int(len(indexes) * .60)
                train_idx_FIR = int(len(indexes) * .87)
                valid_idx_FIR = int(len(indexes) * .90)

                train_indexes_baseline = indexes[:train_idx_baseline]
                valid_indexes_baseline = indexes[train_idx_baseline:valid_idx_baseline]
                train_indexes_FIR = indexes[valid_idx_baseline:train_idx_FIR]
                valid_indexes_FIR= indexes[train_idx_FIR:valid_idx_FIR]
                test_indexes = indexes[valid_idx_FIR:]

                self.train_indexes_BL = train_indexes_baseline
                self.train_indexes_FIR = train_indexes_FIR
                self.valid_indexes_BL = valid_indexes_baseline
                self.valid_indexes_FIR = valid_indexes_FIR
                self.test_indexes = test_indexes

                # expand this shuffling indexing
                for i in range(self.num_classes - 1):
                    self.train_indexes_BL = np.append(self.train_indexes_BL,
                        [x + (i + 1) * self.num_examples_per_class for x in train_indexes_baseline])
                    self.train_indexes_FIR = np.append(self.train_indexes_FIR,
                        [x + (i + 1) * self.num_examples_per_class for x in train_indexes_FIR])
                    self.valid_indexes_BL = np.append(self.valid_indexes_BL,
                        [x + (i + 1) * self.num_examples_per_class for x in valid_indexes_baseline])
                    self.valid_indexes_FIR = np.append(self.valid_indexes_FIR,
                        [x + (i + 1) * self.num_examples_per_class for x in valid_indexes_FIR])
                    self.test_indexes = np.append(self.test_indexes,
                        [x + (i + 1) * self.num_examples_per_class for x in test_indexes])

                # Saving the objects:
                with open('indexes.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                    pkl.dump([self.train_indexes_BL,
                              self.train_indexes_FIR,
                              self.valid_indexes_BL,
                              self.valid_indexes_FIR,
                              self.test_indexes], f)
            else:
                print('File indexes.pkl already exists! I am using the old one')

            print('--------- Indexes check ----------')
            print(len(self.train_indexes_BL) + len(self.train_indexes_FIR) + len(self.valid_indexes_BL) + len(
                self.valid_indexes_FIR) + len(self.test_indexes))


        if self.args.train_cnn:
            print('********************* Generating data for Baseline *********************')
            self.train_generator_BL = DataGenerator(indexes=self.train_indexes_BL,
                                                 batch_size=self.args.batch_size,
                                                 data_path=self.args.data_path, is_2d = self.is_2d)
            self.valid_generator_BL = DataGenerator(indexes=self.valid_indexes_BL,
                                                 batch_size=self.args.batch_size,
                                                 data_path=self.args.data_path, is_2d = self.is_2d)
        if self.args.train_fir:
            print('*********************  Generating data for FIR *********************')
            self.train_generator_FIR = DataGenerator(indexes=self.train_indexes_FIR,
                                                    batch_size=self.args.batch_size,
                                                    data_path=self.args.data_path, is_2d = self.is_2d)
            self.valid_generator_FIR = DataGenerator(indexes=self.valid_indexes_FIR,
                                                    batch_size=self.args.batch_size,
                                                    data_path=self.args.data_path, is_2d = self.is_2d)

        print('*********************  Generating testing data *********************')
        self.test_generator = DataGenerator(indexes=self.test_indexes,
                                            batch_size=self.args.batch_size,
                                            data_path=self.args.data_path, is_2d = self.is_2d)
    

    def train_baseline(self):
        '''Train model through Keras framework.'''
        print('*************** Training Model ***************')
        optimizer = Adam(lr=0.0001)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])
        call_backs = []
        checkpoint = CustomModelCheckpoint(
                os.path.join(self.args.save_path, self.args.bl_model_name),
                monitor='val_acc', verbose=1, save_best_only=True)
        call_backs.append(checkpoint)
        earlystop_callback = EarlyStopping(
                    monitor='val_acc', min_delta=0, patience=self.args.patience,
                    verbose=1, mode='auto')
        call_backs.append(earlystop_callback)

        start_time = time.time()
        self.model.fit_generator(generator=self.train_generator_BL,
                                 steps_per_epoch = self.args.max_steps if self.args.max_steps>0 else None,
                                 epochs=self.args.epochs,
                                 validation_steps=len(self.valid_generator_BL)//self.args.batch_size,
                                 validation_data=self.valid_generator_BL,
                                 shuffle=True,
                                 callbacks=call_backs,
                                 use_multiprocessing=False,
                                 max_queue_size=100)
        train_time = time.time() - start_time
        print('Time to train model %0.3f s' % train_time)
        self.best_model_path = checkpoint.best_path

    def train_FIR(self):
        '''Train model through Keras framework.'''
        print('*************** Training Model ***************')
        optimizer = Adam(lr=0.0001)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])
        call_backs = []
        checkpoint = CustomModelCheckpoint(
            os.path.join(self.args.save_path, self.args.fir_model_name),
            monitor='val_acc', verbose=1, save_best_only=True)
        call_backs.append(checkpoint)
        earlystop_callback = EarlyStopping(
            monitor='val_acc', min_delta=0, patience=self.args.patience,
            verbose=1, mode='auto')
        call_backs.append(earlystop_callback)

        start_time = time.time()
        self.model.fit_generator(generator=self.train_generator_FIR,
                                 steps_per_epoch = self.args.max_steps if self.args.max_steps>0 else None,
                                 epochs=self.args.epochs,
                                 validation_data=self.valid_generator_FIR,
                                 shuffle=True,
                                 callbacks=call_backs,
                                 use_multiprocessing=False,
                                 max_queue_size=100)
        train_time = time.time() - start_time
        print('Time to train model %0.3f s' % train_time)
        self.best_model_path = checkpoint.best_path

    def train_FIR_per_dev(self, dev_id):
        '''Train model through Keras framework.'''
        print('*************** Training Model for Class %d ***************' % dev_id)
        optimizer = Adam(lr=0.0001)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])
        call_backs = []
        checkpoint = CustomModelCheckpoint(
            os.path.join(self.args.save_path, 'per_dev', self.args.fir_model_name.split('.')[0] + '_' + str(dev_id) + '.hdf5'),
            monitor='val_acc', verbose=1, save_best_only=True)
        call_backs.append(checkpoint)
        earlystop_callback = EarlyStopping(
            monitor='val_acc', min_delta=0, patience=self.args.patience,
            verbose=1, mode='auto')
        call_backs.append(earlystop_callback)

        # dataset_index_per_device = dev_id * self.num_examples_per_class + np.array(range(self.num_examples_per_class))
        train_where = np.where(np.logical_and(self.train_indexes_FIR>=dev_id * self.num_examples_per_class, self.train_indexes_FIR < ((dev_id+1) * self.num_examples_per_class-1)))
        valid_indexes_FIR = np.where(np.logical_and(self.valid_indexes_FIR >= dev_id * self.num_examples_per_class, self.valid_indexes_FIR < ((dev_id + 1) * self.num_examples_per_class - 1)))

        print('*********************  Generating data for FIR for Class %d ***************' % dev_id)
        self.train_generator_FIR = DataGenerator(indexes=self.train_indexes_FIR[train_where],
                                                 batch_size=self.args.batch_size,
                                                 data_path=self.args.data_path, is_2d=self.is_2d)
        self.valid_generator_FIR = DataGenerator(indexes=self.valid_indexes_FIR[valid_indexes_FIR],
                                                 batch_size=self.args.batch_size,
                                                 data_path=self.args.data_path, is_2d=self.is_2d)

        start_time = time.time()
        self.model.fit_generator(generator=self.train_generator_FIR,
                                 steps_per_epoch = self.args.max_steps if self.args.max_steps>0 else None,
                                 epochs=self.args.epochs,
                                 validation_data=self.valid_generator_FIR,
                                 shuffle=True,
                                 callbacks=call_backs,
                                 use_multiprocessing=False,
                                 max_queue_size=100)
        train_time = time.time() - start_time
        print('Time to train model %0.3f s' % train_time)
        ## SALVO this must be uncommented before push
        self.best_model_path = checkpoint.best_path

    def test(self, dev_id):
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

        if dev_id < 0:
            optimizer = Adam(lr=0.0001)
            self.model.compile(loss='categorical_crossentropy',
                               optimizer=optimizer,
                               metrics=['accuracy'])
            score = self.model.evaluate_generator(self.test_generator, verbose=1,
                                                  use_multiprocessing = False)
            print(score)
        else:
            optimizer = Adam(lr=0.0001)
            self.model.compile(loss='categorical_crossentropy',
                               optimizer=optimizer,
                               metrics=['accuracy'])

            print('*********************  Generating testing data for Class %d ***************' % dev_id)
            test_indexes_FIR = np.where(np.logical_and(self.test_indexes >= dev_id * self.num_examples_per_class, self.test_indexes < ((dev_id + 1) * self.num_examples_per_class - 1)))

            test_generator_FIR = DataGenerator(indexes=self.test_indexes[test_indexes_FIR],
                                                    batch_size=self.args.batch_size,
                                                    data_path=self.args.data_path, is_2d=self.is_2d)

            ########
            #### compute indexed_per div_id
            score = self.model.evaluate_generator(test_generator_FIR, verbose=1,
                                                  use_multiprocessing=False)
            print(score)

    def run(self):
        '''Run different steps in model pipeline.'''
        if self.args.train_cnn:
            self.build_model_baseline()
            self.load_data()
            self.train_baseline()
            self.test(-1)
        elif self.args.train_fir_perdev:
            for d in range(self.num_classes):
                self.build_model_FIR()
                self.load_data()
                self.train_FIR_per_dev(d)
                self.test(d)
        elif self.args.train_fir:
            self.build_model_FIR()
            self.load_data()
            self.train_FIR()
            self.test(-1)
        else:
            print('EXITING - Please specify model to be trained')


    def parse_arguments(self):
        '''Parse input user arguments.'''
        
        parser = argparse.ArgumentParser(description = 'Train and Validation pipeline',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--max_steps', type=int, default=0,
                            help='Max number of batches. If 0, it uses the whole dataset')

        parser.add_argument('--id_gpu', type=int, default=2,
                            help='GPU to use.')

        parser.add_argument('--bl_model_name', type=str, default='modulation_model.hdf5',
                            help='Name of baseline model.')

        parser.add_argument('--epsilon', type=float, default=0,
                            help='Bound on FIR taps computation in absolute value. If 0, train without bounds.')

        parser.add_argument('--fir_model_name', type=str, default='FIR_model.hdf5',
                            help='Name of baseline model.')

        parser.add_argument('--load_indexes', action='store_true',
                            help='Load indexes from external file. If False, you create and save them in "indexes.pkl".')

        parser.add_argument('--train_cnn', action='store_true',
                            help='Train CNN.')

        parser.add_argument('--train_fir', action='store_true',
                            help='Train FIR one for all.')

        parser.add_argument('--train_fir_perdev', action='store_true',
                            help='Train FIR one per device.')

        parser.add_argument('--num_classes', type=int, default=24,
                            help='Number of classes in the dataset.')

        parser.add_argument('--num_ex_mod', type=int, default=106496,
                            help='Number of examples per class in the dataset.')

        parser.add_argument('--fir_size', type=int, default=10,
                            help='Number of classes in the dataset.')

        parser.add_argument('--save_path', type=str, default='./home/salvo/deepsig_res',
                            help='Path to save weights, model architecture, and logs.')

        parser.add_argument('--h5_path', type=str,
                            default='/mnt/nas/bruno/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5',
                            help='Path to original h5 file.')
        
        #GOLD_XYZ_OSC.0001_1024
        parser.add_argument('--data_path', type=str,
                            default='/mnt/nas/bruno/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5',
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


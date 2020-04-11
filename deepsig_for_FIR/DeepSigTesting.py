import argparse
import h5py
import json
import keras
from keras.models import load_model
import numpy as np
import pickle as pkl
import time
import os

from keras.models import Model
from keras.optimizers import Adam

from keras.utils.io_utils import HDF5Matrix
from DataGenerator import DataGenerator
from DataGeneratorFIR import DataGeneratorFIR
from sklearn.metrics import confusion_matrix

class DeepSigTesting(object):
    '''Class to test model described in paper
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

        # To Do: this is to load the FIR model correctly, might not be useful for other applications, need to un-hardcode this and let users pass it as a parameter
        self.is_2d = True

        self.num_classes = self.args.num_classes

        if self.args.test_single_model or self.args.test_perdev_model:
            self.run()
        else:
            print('You are not testing any model')

    def load_model(self):
        '''Build model architecture.'''
        print('*************** Loading Model ***************')
        if os.path.exists(self.args.model_name):
            self.model = load_model(self.args.model_name)
            self.model.summary()
        else:
            print('Model does not exists')

    def load_testing_data(self):
        '''Load data from path into framework.'''

        print('--------- Loading from File indexes.pkl ---------')

        if os.path.exists(self.args.index_file):
            # Getting back the objects:
            with open('indexes.pkl', 'rb') as f:  # Python 3: open(..., 'rb') note that indexes
                data_loaded = pkl.load(f)

            # To Do: this should be transformed into a dictionary and you pull 'test_indexes only'. Kinda hardcoded to be fixed later
            self.test_indexes = data_loaded[-1]

            print('*********************  Generating testing data *********************')
            self.test_generator = DataGenerator(indexes=self.test_indexes,
                                                batch_size=self.args.batch_size,
                                                data_path=self.args.data_path, is_2d=self.is_2d)

        else:
            print('I have no data to load, please give me data (e.g., indexes.pkl)')

    def load_testing_data_for_FIR(self):
        '''Load data from path into framework.'''

        print('--------- Loading from File indexes.pkl ---------')

        if os.path.exists(self.args.index_file):
            # Getting back the objects:
            with open('indexes.pkl', 'rb') as f:  # Python 3: open(..., 'rb') note that indexes
                data_loaded = pkl.load(f)

            # To Do: this should be transformed into a dictionary and you pull 'test_indexes' only. Kinda hardcoded to be fixed later
            self.test_indexes = data_loaded[-1]

            print('*********************  Generating testing data *********************')
            self.test_generator = DataGeneratorFIR(indexes=self.test_indexes,
                                                batch_size=self.args.batch_size,
                                                data_path=self.args.data_path, is_2d=self.is_2d,
                                                models_path = self.args.models_path,
                                                model_name = self.args.fir_model_name,
                                                taps_name = self.args.taps_name,
                                                FIR_layer_name = self.args.fir_layer_name,
                                                num_classes = self.num_classes)

        else:
            print('I have no data to load, please give me data (e.g., indexes.pkl)')

    def get_predicted_label(self,labels):
        unique, counts = np.unique(labels, return_counts=True)
        predicted_label = unique[np.argmax(counts)]
        return predicted_label

    def test_model(self):
        optimizer = Adam(lr=0.0001)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])

        # score_eval = self.model.evaluate_generator(self.test_generator, verbose=1, use_multiprocessing = False)
        # print(score_eval)
        score_predict = self.model.predict_generator(self.test_generator, verbose=1, use_multiprocessing = False)
        label_predict = np.argmax(score_predict,1)

        Y_real = HDF5Matrix(self.args.data_path, 'Y')
        label_true = np.zeros(label_predict.shape)
        for i in range(label_predict.shape[0]):
            # print('Index : ', self.test_indexes[i])
            # print(Y_real.__getitem__(self.test_indexes[i]))
            # print(np.argmax(Y_real.__getitem__(self.test_indexes[i])))
            label_true[i] = np.argmax(Y_real.__getitem__(self.test_indexes[i]))

        con_matrix = confusion_matrix(label_true, label_predict)
        con_matrix_perc = con_matrix/ con_matrix.astype(np.float).sum(axis=1)
        example_accuracy = np.mean(np.diag(con_matrix_perc))

        # compute batch accuracy
        num_batches = int(self.test_indexes.shape/self.args.batch_size)
        batch_prediction_indicator = np.zeros([num_batches,])

        for b in range(num_batches):
            true_batch_label = label_true[b*self.args.batch_size] # label is the same for each batch, you can just take the first element
            predicted_batch_label = self.get_predicted_label(label_predict[b*self.args.batch_size:(b+1)*self.args.batch_size])
            if true_batch_label == predicted_batch_label:
                batch_prediction_indicator[b] = 1

        batch_accuracy = np.mean(batch_prediction_indicator)

        my_dict = {'example_accuracy' : example_accuracy,
                   'batch_accuracy': batch_accuracy,
                   'confusion_matrix' : con_matrix_perc}

        # Saving the objects:
        save_name = os.path.join(self.args.save_path,(self.args.save_file_name + '.pkl'))
        with open(save_name, 'wb') as f:  # Python 3: open(..., 'wb')
            pkl.dump(my_dict, f)


    def run(self):
        '''Run different steps in model pipeline.'''
        if self.args.test_single_model: # This is used to test the same model for all classes
            self.load_model()
            self.load_testing_data()
            self.test_model()
        elif self.args.test_perdev_model: # this is used to test same architecture, but FIRs are different for each class
            self.load_model()
            self.load_testing_data_for_FIR()
            self.test_model()
        else:
            print('EXITING - Please specify model to be tested')

    def parse_arguments(self):
        '''Parse input user arguments.'''

        parser = argparse.ArgumentParser(description='Testing-only pipeline',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--id_gpu', type=int, default=2,
                            help='GPU to use.')

        parser.add_argument('--model_name', type=str, default='./home/salvo/deepsig_res/modulation_model.hdf5',
                            help='Name of baseline model.')

        parser.add_argument('--fir_model_name', type=str, default='FIR_model',
                            help='Name of the FIR model without the hdf5 extension, just the prefix used to generate them.')

        parser.add_argument('--taps_name', type=str, default='phi',
                            help='Name of the variable storing the taps.')

        parser.add_argument('--fir_layer_name', type=str, default='FIR_layer',
                            help='Name of the FIR layer in Keras, this is usually FIR_layer when you create the model.')

        parser.add_argument('--models_path', type=str, default='./home/salvo/deepsig_res/per_dev',
                            help='Path where all FIR models are saved.')

        parser.add_argument('--index_file', type=str, default = 'indexes.pkl',
                            help='Name of index pickle file, usually we store a dictionary "indexes.pkl" with indexes in there. MUST contain a "test_indexes" variable')

        parser.add_argument('--test_single_model', action='store_true',
                            help='Test either baseline, or unique FIR+Baseline CNN.')

        parser.add_argument('--test_perdev_model', action='store_true',
                            help='Test one FIR one for each dev with frozen bas.')

        parser.add_argument('--num_classes', type=int, default=24,
                            help='Number of classes in the dataset.')

        parser.add_argument('--max_steps', type=int, default=0,
                            help='Max number of batches. If 0, it uses the whole dataset')

        parser.add_argument('--save_path', type=str, default='./home/salvo/deepsig_res',
                            help='Path to save weights, model architecture, and logs.')

        parser.add_argument('--save_file_name', type=str, default='results.pkl',
                            help='File name to save, results are saved in save_path.')

        parser.add_argument('--data_path', type=str,
                            default='/mnt/nas/bruno/deepsig/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5',
                            help='Path to data.')

        parser.add_argument('--batch_size', type=int, default=32,
                            help='Batch size for model optimization.')

        return parser.parse_args()

if __name__ == '__main__':
    DeepSigTesting()
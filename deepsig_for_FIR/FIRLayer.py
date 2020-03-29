import numpy as np
import tensorflow as tf
from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Conv1D
from keras.initializers import glorot_normal
from FIRInitializer import FIRInitializer


class FIRLayer(Layer):
    '''
    Finite Impulse Response filter layer. This layer performs convolutions on
    complex input data.
    '''

    def __init__(self, filter_dim=11, channels=1, strides=1, verbose=0, output_dim=None,
        kernel_regularizer=None, **kwargs):

        self.filter_dim = filter_dim
        self.channels = channels
        self.strides = strides
        self.output_dim = output_dim
        self.verbose = verbose
        self.kernel_regularizer=kernel_regularizer
        super(FIRLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        '''Define the layer weights.'''
        self.phi = self.add_weight(name='phi',
                                   shape=(self.filter_dim, input_shape[2], self.channels), # (5, 2, 1)
                                   initializer=FIRInitializer(),
                                   regularizer=self.kernel_regularizer,
                                   trainable=True)
        '''Define layer output length'''
        self.output_dim = int(np.ceil(1.*input_shape[1]/self.strides))
        if self.verbose:
            print('input: ', input_shape) # (?, N, 2)
            print('filter_dim: ', self.filter_dim)
            print('output: ', self.output_dim)
            print('phi: ', self.phi.shape)

        super(FIRLayer, self).build(input_shape)


    def call(self, X):
        '''
        Convolution of complex numbers logic.
            X : (?, N, 2)
            N : (M, 2, 1)
            X is reshaped to X_real = (?, N, 1)    
                             X_img = (?, N, 1)

            phi is reshaped to phi_real = (M, 1, 1)
                               phi_img = (M, 1, 1)
        '''
        X = K.expand_dims(X, 3) # (?, N, 2, 1)
       
        X_real = X[:, :, 0] # (?, N, 1)
        X_img = X[:, :, 1] # (?, N, 1)

        phi_real = self.phi[:, 0, :] # (M, 1)
        phi_real = K.expand_dims(phi_real, 1) # (M, 1, 1)

        phi_img = self.phi[:, 1, :] # (M, 1)
        phi_img = K.expand_dims(phi_img, 1) # (M, 1, 1)

        real = K.conv1d(X_real, phi_real, strides=self.strides, padding='same', data_format='channels_last') - K.conv1d(X_img, phi_img, strides=self.strides, padding='same', data_format='channels_last')
        img = K.conv1d(X_real, phi_img, strides=self.strides, padding='same', data_format='channels_last') + K.conv1d(X_img, phi_real, strides=self.strides, padding='same', data_format='channels_last')

        real = K.squeeze(real, 2)
        img = K.squeeze(img, 2)

        X_filtered = K.stack([real, img])
        X_filtered = K.permute_dimensions(X_filtered, (1, 2, 0))

        if self.verbose:
            print( 'X: ', X.shape)
            print('self.phi: ', self.phi.shape)
            print('X_real: ', X_real.shape)
            print('X_img: ', X_img.shape)
            print('phi_real: ', phi_real.shape)
            print('phi_img: ', phi_img.shape)
            print('X_filtered: ', X_filtered.shape)

        return X_filtered
         

    def compute_output_shape(self, input_shape):
        '''
        In case the layer modifies the shape of its input, do automatic shape inference. 
        '''
        return (input_shape[0], self.output_dim, input_shape[2])

    def get_config(self):
        config = {
            'filter_dim': self.filter_dim,
            'channels': self.channels,
            'strides': self.strides,
            'output_dim': self.output_dim,
            'verbose': self.verbose
        }

        base_config = super(FIRLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items())) 

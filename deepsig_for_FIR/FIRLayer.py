import numpy as np
import tensorflow as tf
from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Conv1D
from keras.initializers import glorot_normal
from FIRInitializer import FIRInitializer
from FIRConstraint import FIRConstraint

class FIRLayer(Layer):
    '''
    Finite Impulse Response filter layer. This layer performs convolutions on
    complex input data.
    '''

    def __init__(self, filter_dim=11, channels=1, strides=1, verbose=0, output_dim=None,
                 kernel_regularizer=None, epsilon=None, **kwargs):

        self.filter_dim = filter_dim
        self.channels = channels
        self.strides = strides
        self.output_dim = output_dim
        self.verbose = verbose
        self.epsilon=epsilon
        self.kernel_regularizer = kernel_regularizer
        super(FIRLayer, self).__init__(**kwargs)
        if self.verbose:
            print('Inside __init__ : output shape ', self.output_dim)  # OK (1, 1024, 2)
            for key, value in kwargs.items():
                print("Inside __init__ : %s == %s" % (key, value))  # OK (1, 1024, 2)

    def build(self, input_shape):
        '''Define the layer weights.'''
        self.actual_input_shape = input_shape

        print(self.actual_input_shape)

        self.phi = self.add_weight(name='phi',                        # here was input_shape[2], To do: if we want to have filters taps with more dimensions, this is where we should get rid of the [-1]
                                   shape=(self.filter_dim, self.actual_input_shape[-1], self.channels),  # (5, 2, 1)
                                   initializer=FIRInitializer(),
                                   regularizer=self.kernel_regularizer,
                                   trainable=True,
                                   constraint=FIRConstraint(epsilon=self.epsilon))
        '''Define layer output length'''
        self.output_dim = int(np.ceil(1. * self.actual_input_shape[1] / self.strides))
        if self.verbose:
            print('Inside build() -- input: ', input_shape)  # (?, 1, N, 2))
            print('Inside build() -- actual input: ', self.actual_input_shape) # (?, N, 2))
            print('Inside build() -- filter_dim: ', self.filter_dim)
            print('Inside build() -- output: ', self.output_dim)
            print('Inside build() -- phi: ', self.phi.shape)

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

        X_temp = X

        # input_shape is here (?, 1, 1024, 2), we are removing the dimension == 1 ---> becomes (?, 1024, 2)
        trivial_dimension = np.argwhere(np.array(np.shape(X)) == 1)[0]
        if len(trivial_dimension):  # this is only if we need to remove dummy dimensions
            X = tf.squeeze(X, axis=int(trivial_dimension))
            if self.verbose:
                print('Inside call() -- Squeeze dimension %f in FIRlayer' % int(trivial_dimension))
                print('Inside call() -- X.len() is ', np.shape(X))
                print('Inside call() -- X_temp.len() is ', np.shape(X_temp))

        X = K.expand_dims(X, 3)  # (?, N, 2, 1)

        X_real = X[:, :, 0]  # (?, N, 1)
        X_img = X[:, :, 1]  # (?, N, 1)

        phi_real = self.phi[:, 0, :]  # (M, 1)
        phi_real = K.expand_dims(phi_real, 1)  # (M, 1, 1)

        phi_img = self.phi[:, 1, :]  # (M, 1)
        phi_img = K.expand_dims(phi_img, 1)  # (M, 1, 1)

        real = K.conv1d(X_real, phi_real, strides=self.strides, padding='same', data_format='channels_last') - K.conv1d(
            X_img, phi_img, strides=self.strides, padding='same', data_format='channels_last')
        img = K.conv1d(X_real, phi_img, strides=self.strides, padding='same', data_format='channels_last') + K.conv1d(
            X_img, phi_real, strides=self.strides, padding='same', data_format='channels_last')

        real = K.squeeze(real, 2)
        img = K.squeeze(img, 2)

        X_filtered = K.stack([real, img])
        X_filtered = K.permute_dimensions(X_filtered, (1, 2, 0))

        # add trivial dimension back
        if len(trivial_dimension):
            X_filtered = K.expand_dims(X_filtered, int(trivial_dimension))  # (?, 1, N, 2)
            if self.verbose:
                print('Inside call() -- Expand dimension %f in FIRlayer' % int(trivial_dimension))

        if self.verbose:
            print('Inside call() -- X_filtered.len() is ', np.shape(X_filtered))
            print('Inside call() -- X_temp.len() is ', np.shape(X_temp))
            print('Inside call() -- X.len(): ', X.shape)
            print('Inside call() -- self.phi.len(): ', self.phi.shape)
            print('Inside call() -- X_real.len(): ', X_real.shape)
            print('Inside call() -- X_img.len(): ', X_img.shape)
            print('Inside call() -- phi_real.len(): ', phi_real.shape)
            print('Inside call() -- phi_img.len(): ', phi_img.shape)

        return X_filtered

#    def compute_output_shape(self, input_shape):
#        '''
#        In case the layer modifies the shape of its input, do automatic shape inference.
#       '''
#      return (input_shape[0], 1, self.output_dim, input_shape[3])

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
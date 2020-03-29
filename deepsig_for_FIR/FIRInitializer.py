import numpy as np
import keras.backend as K
from keras.initializers import Initializer

class FIRInitializer(Initializer):
    '''
    Initializer that generates a tensor of [1, 0, 0, ..., N] +- random gaussian noise.
    '''
    
    def __call__(self, shape, dtype=None):
#        print 'FIRInitializer shape: ', shape
        scale = np.sqrt(1.0/shape[0])
        #tensor = np.zeros(shape=shape)
        tensor = np.random.normal(loc=0.0, scale=scale, size=shape)
        tensor[0][0] = 1
        tensor = K.variable(tensor)
        #print 'FIR init weight: ', K.eval(tensor)
        return tensor

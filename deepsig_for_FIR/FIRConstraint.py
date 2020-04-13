from keras.constraints import Constraint
import keras.backend as K
import numpy as np

class FIRConstraint(Constraint):

    def __init__(self, epsilon=0.1, shape=None):
        self.epsilon = epsilon
        self.base_fir = np.zeros(shape)
        self.base_fir[0][0] = 1
        self.base_fir = K.variable(self.base_fir)

    def __call__(self, w):
        return K.clip(w-self.base_fir, min_value=-self.epsilon, max_value=self.epsilon) + self.base_fir

    def get_config(self):
        return {'epsilon': self.epsilon}

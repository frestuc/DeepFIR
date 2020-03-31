from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Flatten, Reshape, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, MaxPooling1D, Conv1D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import backend as K
import numpy as np

# K.clear_session()

inputs = Input(shape=(1, 1024, 2), name = 'Input')
x = Conv2D(2, kernel_size=1, name="bananefritteLOL")(inputs)
# x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name = 'krustyilclown')(x)
x = Conv2D(64, kernel_size=1, name = 'Conv_1')(x)
x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name = 'MaxPool_1')(x)
x = Conv2D(64, kernel_size=1, name = 'Conv_2')(x)
x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name = 'MaxPool_2')(x)
x = Conv2D(64, kernel_size=1, name = 'Conv_3')(x)
x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name = 'MaxPool_3')(x)
x = Conv2D(64, kernel_size=1, name = 'Conv_4')(x)
x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name = 'MaxPool_4')(x)
x = Conv2D(64, kernel_size=1, name = 'Conv_5')(x)
x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name = 'MaxPool_5')(x)
x = Conv2D(64, kernel_size=1, name = 'Conv_6')(x)
x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name = 'MaxPool_6')(x)
x = Conv2D(64, kernel_size=1, name = 'Conv_7')(x)
x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name = 'MaxPool_7')(x)
x = Flatten(name = 'Flatten')(x)
x = Dense(128, activation='selu', name = 'Dense_1')(x)
x = Dense(128, activation='selu', name = 'Dense_2')(x)
x = Dense(24, activation='softmax', name = 'Softmax')(x)
model1 = Model(inputs=inputs, outputs=x)
model1.summary()

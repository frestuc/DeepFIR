from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Flatten, Reshape, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, MaxPooling1D, Conv1D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import backend as K

model_name = 'C:\\Users\\totix\\Desktop\\darpa\\modulation_model.hdf5'

K.clear_session()
inputs = Input(shape=(1, 1024, 2))
x = Conv2D(64, kernel_size=1, name="bananefritteLOL")(inputs)
x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name = 'krustyilclown')(x)
x = Conv2D(64, kernel_size=1)(x)
x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last')(x)
x = Conv2D(64, kernel_size=1)(x)
x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last')(x)
x = Conv2D(64, kernel_size=1)(x)
x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last')(x)
x = Conv2D(64, kernel_size=1)(x)
x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last')(x)
x = Conv2D(64, kernel_size=1)(x)
x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last')(x)
x = Conv2D(64, kernel_size=1)(x)
x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last')(x)
x = Conv2D(64, kernel_size=1)(x)
x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last')(x)
x = Flatten()(x)
x = Dense(128, activation='selu')(x)
x = Dense(128, activation='selu')(x)
x = Dense(24, activation='softmax')(x)


model1 = Model(inputs=inputs, outputs=x)
model1.summary()

# load model
K.clear_session()
model2 = load_model(model_name)
model2.summary()

K.clear_session()
model2.load_weights(model_name,by_name = True, skip_mismatch=True)

# load weights
K.clear_session()
model1.load_weights(model_name,by_name = True, skip_mismatch=True)
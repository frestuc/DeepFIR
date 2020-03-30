from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Flatten, Reshape, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, MaxPooling1D, Conv1D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import backend as K
import numpy as np

# model_name = 'C:\\Users\\totix\\Desktop\\darpa\\modulation_model.hdf5'
model_name = 'test.hdf5'

# inputs = Input(shape=(1, 1024, 2), name = 'Input')
# x = Conv2D(64, kernel_size=1, name = 'Conv_1')(inputs)
# x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name = 'MaxPool_1')(x)
# x = Conv2D(64, kernel_size=1, name = 'Conv_2')(x)
# x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name = 'MaxPool_2')(x)
# x = Conv2D(64, kernel_size=1, name = 'Conv_3')(x)
# x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name = 'MaxPool_3')(x)
# x = Conv2D(64, kernel_size=1, name = 'Conv_4')(x)
# x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name = 'MaxPool_4')(x)
# x = Conv2D(64, kernel_size=1, name = 'Conv_5')(x)
# x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name = 'MaxPool_5')(x)
# x = Conv2D(64, kernel_size=1, name = 'Conv_6')(x)
# x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name = 'MaxPool_6')(x)
# x = Conv2D(64, kernel_size=1, name = 'Conv_7')(x)
# x = MaxPooling2D(pool_size=(1, 2), data_format='channels_last', name = 'MaxPool_7')(x)
# x = Flatten(name = 'Flatten')(x)
# x = Dense(128, activation='selu', name = 'Dense_1')(x)
# x = Dense(128, activation='selu', name = 'Dense_2')(x)
# x = Dense(24, activation='softmax', name = 'Softmax')(x)
# model2 = Model(inputs=inputs, outputs=x)
# optimizer = Adam(lr=0.0001)
# model2.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# model2.summary()
# # model2.save_weights(model_name,overwrite=True)
# model2.save(model_name,overwrite=True)

# load model
# K.clear_session()
model2 = load_model(model_name)
model2.summary()


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

for i in range(len(model2.layers)-3):
    print(model1.layers[i+2].name, model2.layers[i+1].name)
    print(np.sum(np.sum(np.sum(np.sum(model1.layers[i+2].get_weights())))) ,
          np.sum(np.sum(np.sum(np.sum(model2.layers[i+1].get_weights())))))


# K.clear_session()
model1.load_weights(model_name, by_name = True, skip_mismatch = True, reshape = False)

#  print for loop with sum weight per layer and compare

for i in range(len(model2.layers)-3):
    print(model1.layers[i+2].name, model2.layers[i+1].name)
    print(np.sum(np.sum(np.sum(np.sum(model1.layers[i+2].get_weights())))) ,
          np.sum(np.sum(np.sum(np.sum(model2.layers[i+1].get_weights())))))

print('missing')
print(np.sum(np.sum(model1.layers[1].get_weights())))
# print(np.sum(np.sum(model1.layers[2].get_weights())))
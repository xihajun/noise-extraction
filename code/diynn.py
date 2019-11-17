import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, UpSampling2D, Cropping2D, Input, Flatten, Dense, BatchNormalization, Dropout
from keras.models import Model
from keras import backend as K

width,height,depth= 201,201,3

def autoencoder(input_tensor=None, trainable=False):

    input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3

    # Block 1
    x = Conv2D(32,kernel_size = (2,2), 
                input_shape = (width,height,depth), 
                activation = 'relu',
                padding = 'same')(img_input)
    x = MaxPooling2D(pool_size = (2,2), padding = 'same')(x)

    # Block 2
    x = Conv2D(32,
                kernel_size = (2,2), 
                activation = 'relu',
                padding = 'same')(x)
    x = MaxPooling2D(pool_size = (2,2), padding = 'same')(x)

    # Block 3
    x = Conv2D(32,
                kernel_size = (2,2), 
                activation = 'relu',
                padding = 'same')(x)
    x = MaxPooling2D(pool_size = (2,2), padding = 'same')(x)

    # Upsampling 1
    x = UpSampling2D(size=(2, 2),
                     data_format=None,
                     interpolation='nearest')(x)
    x = Conv2D(16, kernel_size = (2,2), 
               activation = 'relu')(x)

    # Upsampling 2
    x = UpSampling2D(size=(2, 2),
                     data_format=None,
                     interpolation='nearest')(x)
    x = Conv2D(16, kernel_size = (2,2), 
               activation = 'relu')(x)
    
    # Upsampling 3
    x = UpSampling2D(size=(2, 2),
                     data_format=None,
                     interpolation='nearest')(x)
    x = Conv2D(3, kernel_size = (2,2), 
               activation = 'relu')(x)

    return x
    
def CNN(input_tensor=None, trainable=False):

    input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3

    # Block 1
    x = Conv2D(64,kernel_size = (5,5), 
                 data_format='channels_last',
                input_shape = (width,height,depth), 
                activation = 'relu',
                padding = 'same')(img_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (2,2), padding = 'same')(x)

    # Block 2
    x = Conv2D(64,kernel_size = (5,5), 
                 data_format='channels_last',
                input_shape = (width,height,depth), 
                activation = 'relu',
                padding = 'same')(img_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (2,2), padding = 'same')(x)

    # Block 3
    x = Conv2D(64,kernel_size = (5,5), 
                 data_format='channels_last',
                input_shape = (width,height,depth), 
                activation = 'relu',
                padding = 'same')(img_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (2,2), padding = 'same')(x)

    # Flatten
    x = Flatten()(x)
    # Dropout
#    x = Dropout(0.1)(x)
    # Dense NN
    x = Dense(32,activation='relu')(x)
    x = Dense(32,activation='relu')(x)
    x = Dense(32,activation='relu')(x)
    x = Dense(32,activation='relu')(x)

    # classification
    x = Dense(1,activation='sigmoid')(x)
    
    return x
    
    

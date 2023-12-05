import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Input, GlobalAveragePooling2D, AveragePooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.callbacks import TensorBoard
import datetime

def InceptionV3(input_layer):
    x = StemBlock(input_layer)
    x = InceptionBlock_A(prev_layer = x , nbr_kernels = 16)
    x = InceptionBlock_A(prev_layer = x , nbr_kernels = 32)
    x = ReductionBlock_A(prev_layer = x)
    x = InceptionBlock_B(prev_layer = x  , nbr_kernels = 64)
    x = InceptionBlock_B(prev_layer = x , nbr_kernels = 96)
    Aux = auxiliary_classifier(prev_Layer = x)
    x = ReductionBlock_B(prev_layer = x)
    x = InceptionBlock_C(prev_layer = x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=512, activation='relu')(x)
    x = Dropout(rate=0.3)(x)
    x = Dense(units=10, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=[x, Aux], name='Inception-V3-CIFAR10')
    return model

def conv_with_Batch_Normalisation(prev_layer , nbr_kernels , filter_Size , strides =(1,1) , padding = 'same'):
    x = Conv2D(filters=nbr_kernels, kernel_size = filter_Size, strides=strides , padding=padding)(prev_layer)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation='relu')(x)
    return x

def StemBlock(prev_layer):
    x = conv_with_Batch_Normalisation(prev_layer, nbr_kernels = 16, filter_Size=(3,3))
    x = conv_with_Batch_Normalisation(x, nbr_kernels = 32, filter_Size=(3,3))
    x = conv_with_Batch_Normalisation(x, nbr_kernels = 64, filter_Size=(3,3))
    x = MaxPool2D(pool_size=(2,2) , strides=(2,2))(x)
    x = conv_with_Batch_Normalisation(x, nbr_kernels = 80, filter_Size=(1,1))
    return x

def InceptionBlock_A(prev_layer  , nbr_kernels):
    branch1 = conv_with_Batch_Normalisation(prev_layer, nbr_kernels = 32, filter_Size = (1,1))
    branch1 = conv_with_Batch_Normalisation(branch1, nbr_kernels=48, filter_Size=(3,3))
    branch1 = conv_with_Batch_Normalisation(branch1, nbr_kernels=48, filter_Size=(3,3))
    branch2 = conv_with_Batch_Normalisation(prev_layer, nbr_kernels=24, filter_Size=(1,1))
    branch2 = conv_with_Batch_Normalisation(branch2, nbr_kernels=32, filter_Size=(3,3))
    branch3 = AveragePooling2D(pool_size=(3,3) , strides=(1,1) , padding='same') (prev_layer)
    branch3 = conv_with_Batch_Normalisation(branch3, nbr_kernels = nbr_kernels, filter_Size = (1,1))
    branch4 = conv_with_Batch_Normalisation(prev_layer, nbr_kernels=32, filter_Size=(1,1))
    output = concatenate([branch1 , branch2 , branch3 , branch4], axis=3)
    return output

def InceptionBlock_B(prev_layer , nbr_kernels):
    branch1 = conv_with_Batch_Normalisation(prev_layer, nbr_kernels = nbr_kernels, filter_Size = (1,1))
    branch1 = conv_with_Batch_Normalisation(branch1, nbr_kernels = nbr_kernels, filter_Size = (7,1))
    branch1 = conv_with_Batch_Normalisation(branch1, nbr_kernels = nbr_kernels, filter_Size = (1,7))
    branch1 = conv_with_Batch_Normalisation(branch1, nbr_kernels = 96, filter_Size = (1,7))
    branch2 = conv_with_Batch_Normalisation(prev_layer, nbr_kernels = nbr_kernels, filter_Size = (1,1))
    branch2 = conv_with_Batch_Normalisation(branch2, nbr_kernels = nbr_kernels, filter_Size = (1,7))
    branch2 = conv_with_Batch_Normalisation(branch2, nbr_kernels = 96, filter_Size = (7,1))
    branch3 = AveragePooling2D(pool_size=(3,3) , strides=(1,1) , padding ='same') (prev_layer)
    branch3 = conv_with_Batch_Normalisation(branch3, nbr_kernels = 96, filter_Size = (1,1))
    branch4 = conv_with_Batch_Normalisation(prev_layer, nbr_kernels = 96, filter_Size = (1,1))
    output = concatenate([branch1 , branch2 , branch3 , branch4], axis = 3)
    return output

def InceptionBlock_C(prev_layer):
    branch1 = conv_with_Batch_Normalisation(prev_layer, nbr_kernels = 224, filter_Size = (1,1))
    branch1 = conv_with_Batch_Normalisation(branch1, nbr_kernels = 192, filter_Size = (3,3))
    branch1_1 = conv_with_Batch_Normalisation(branch1, nbr_kernels = 192, filter_Size = (1,3))
    branch1_2 = conv_with_Batch_Normalisation(branch1, nbr_kernels = 192, filter_Size = (3,1))
    branch1 = concatenate([branch1_1 , branch1_2], axis = 3)
    branch2 = conv_with_Batch_Normalisation(prev_layer, nbr_kernels = 192, filter_Size = (1,1))
    branch2_1 = conv_with_Batch_Normalisation(branch2, nbr_kernels = 192, filter_Size = (1,3))
    branch2_2 = conv_with_Batch_Normalisation(branch2, nbr_kernels = 192, filter_Size = (3,1))
    branch2 = concatenate([branch2_1 , branch2_2], axis = 3)
    branch3 = AveragePooling2D(pool_size=(3,3) , strides=(1,1) , padding='same')(prev_layer)
    branch3 = conv_with_Batch_Normalisation(branch3, nbr_kernels = 96, filter_Size = (1,1))
    branch4 = conv_with_Batch_Normalisation(prev_layer, nbr_kernels = 160, filter_Size = (1,1))
    output = concatenate([branch1 , branch2 , branch3 , branch4], axis = 3)
    return output

def ReductionBlock_A(prev_layer):
    branch1 = conv_with_Batch_Normalisation(prev_layer, nbr_kernels = 32, filter_Size = (1,1))
    branch1 = conv_with_Batch_Normalisation(branch1, nbr_kernels = 48, filter_Size = (3,3))
    branch1 = conv_with_Batch_Normalisation(branch1, nbr_kernels = 48, filter_Size = (3,3) , strides=(2,2) , padding='valid')
    branch2 = conv_with_Batch_Normalisation(prev_layer, nbr_kernels = 192, filter_Size=(3,3) , strides=(2,2) , padding='valid')
    branch3 = MaxPool2D(pool_size=(3,3) , strides=(2,2) , padding='valid')(prev_layer)
    output = concatenate([branch1 , branch2 , branch3], axis = 3)
    return output

def ReductionBlock_B(prev_layer):
    branch1 = conv_with_Batch_Normalisation(prev_layer, nbr_kernels = 96, filter_Size = (1,1))
    branch1 = conv_with_Batch_Normalisation(branch1, nbr_kernels = 96, filter_Size = (1,7))
    branch1 = conv_with_Batch_Normalisation(branch1, nbr_kernels = 96, filter_Size = (7,1))
    branch1 = conv_with_Batch_Normalisation(branch1, nbr_kernels = 96, filter_Size = (3,3) , strides=(2,2) , padding = 'valid')
    branch2 = conv_with_Batch_Normalisation(prev_layer, nbr_kernels = 96, filter_Size = (1,1) )
    branch2 = conv_with_Batch_Normalisation(branch2, nbr_kernels = 160, filter_Size = (3,3) , strides=(2,2) , padding='valid' )
    branch3 = MaxPool2D(pool_size=(3,3) , strides=(2,2) , padding='valid')(prev_layer)
    output = concatenate([branch1 , branch2 , branch3], axis = 3)
    return output

def auxiliary_classifier(prev_Layer):
    x = AveragePooling2D(pool_size=(5,5) , strides=(3,3)) (prev_Layer)
    x = conv_with_Batch_Normalisation(x, nbr_kernels = 64, filter_Size = (1,1))
    x = Flatten()(x)
    x = Dense(units = 256, activation='relu') (x)
    x = Dropout(rate = 0.3) (x)
    x = Dense(units = 10, activation='softmax') (x)
    return x

x_tensor = Input(shape=(32, 32, 3))
model = InceptionV3(x_tensor)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy'],
    loss_weights=[1.0, 0.4],
    metrics=['accuracy']
)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

history = model.fit(
    x_train, [y_train, y_train],
    batch_size=32,
    epochs=50,
    validation_data=(x_test, [y_test, y_test]),
    callbacks=[tensorboard_callback]
)
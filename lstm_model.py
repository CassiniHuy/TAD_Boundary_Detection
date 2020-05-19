# From https://github.com/lincshunter/TADBoundaryDectector

import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras import models

# network and training
NB_EPOCH = 150
BATCH_SIZE = 50
VERBOSE = 1
NB_CLASSES = 2 # number of classes
METRICS =['acc']
LOSS = 'binary_crossentropy'
KERNEL_INITIAL ='glorot_uniform'

def three_CNN_LSTM(learning_rate=0.001,INPUT_SHAPE=[1000, 4],KERNEL_SIZE=9,NUM_KERNEL=64,RNN_UNITS=40):
    params = locals()

    inp = Input(shape=INPUT_SHAPE)
    x = layers.Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL)(inp)
    x = layers.Activation('relu')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.MaxPooling1D()(x)

    x = layers.Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL)(x)
    x = layers.Activation('relu')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPooling1D()(x)
    
    x = layers.Conv1D(NUM_KERNEL,kernel_size=KERNEL_SIZE,kernel_initializer=KERNEL_INITIAL)(x)
    x = layers.Activation('relu')(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPooling1D()(x)

    #LSTM
    #HIDDEN_UNITS = 20
    x = layers.Bidirectional(layers.LSTM(RNN_UNITS,kernel_initializer=KERNEL_INITIAL,return_sequences=True,dropout=0.5))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    #a soft max classifier
    x = layers.Activation('sigmoid')(x)
    
    return models.Model(inp, x), params
  


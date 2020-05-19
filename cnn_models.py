#!/usr/bin/env python
# coding: utf-8

# Author: SCU Wuyuzhang College, Wei Chengan & Wan Ziyi
# Data: 2020/5

import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras import regularizers
from tensorflow.keras import models

from lstm_model import three_CNN_LSTM

def tokenize(x, n, padding='same'):
    Y = K.argmax(x, axis=-1)
    if n == 1:
        return Y
    W = K.constant([[[4**k]] for k in range(n)])
    Y = K.cast(Y, dtype=tf.float32)
    W = K.cast(W, dtype=tf.float32)
    Y = K.expand_dims(Y, axis=-1)
    Y = K.conv1d(x=Y, kernel=W, strides=1, padding=padding)
    Y = K.squeeze(Y, axis=-1)
    return Y

def one_hot_layer(x, n):
    return tf.one_hot(indices=tf.to_int32(x), depth=n, on_value=1.0, off_value=0.0, axis=-1)

def reverse_seq(x):
    return tf.reverse_v2(x, axis=[1])

''' Model1: CNN '''
def model_cnn(
    inp_shape=[1000, 4], 
    embed_n=4,
    embed_dim=64, 
    cnn_filters=[32], 
    cnn_kernels=[2],
    cnn_dilations=[[1, 1, 1]],
    cnn_dropouts=[0.5],
    cnn_regularizers=None, 
    pooling='local',
    max_pool=2,
    dense_regularizer=None,
    batchnormal=False):

    hparams = locals()
    ''' Input layer '''
    inp = Input(shape=inp_shape)

    ''' Embedding layer or not '''
    if embed_dim > 0:
        x = layers.Lambda(tokenize, arguments={'n': embed_n, 'padding': 'valid'})(inp)
        x = layers.Embedding(4**embed_n, embed_dim, input_length=[1000 - embed_n + 1])(x)
    else:
        if embed_n == 1:
            x = inp
        else:
            x = layers.Lambda(tokenize, arguments={'n': embed_n, 'padding': 'valid'})(inp)
            x = layers.Lambda(one_hot_layer, arguments={'n': 4**embed_n})(x)

    ''' Conv layers '''

    xs = [None] * len(cnn_kernels)
    if cnn_regularizers == None:
        cnn_regularizers = [None]*len(cnn_kernels)

    for i in range(len(cnn_kernels)):
        xs[i] = x
        for dil in cnn_dilations[i]:
            xs[i] = layers.Conv1D(
                cnn_filters[i], 
                cnn_kernels[i], 
                kernel_regularizer=cnn_regularizers[i], 
                dilation_rate=dil)(xs[i])
            if batchnormal:
                xs[i] = layers.BatchNormalization(axis=-1, momentum=0.9)(xs[i], training=True)
            xs[i] = layers.Activation('relu')(xs[i])
            xs[i] = layers.Dropout(cnn_dropouts[i])(xs[i])
            xs[i] = layers.MaxPooling1D(pool_size=max_pool)(xs[i])
        
        if pooling == 'global':
            xs[i] = layers.GlobalMaxPooling1D()(xs[i])
        elif pooling == 'local':
            xs[i] = layers.Flatten()(xs[i])
        else:
            raise Exception('Unsupported param: {0}'.format(pooling))

    ''' Concatenate all vectors into one '''
    if len(xs) == 1:
        x = xs[0]
    else:
        x = layers.concatenate(xs, axis=-1)

    ''' Dense layer '''
    x = layers.Dense(1, kernel_regularizer=dense_regularizer)(x)
    outp = layers.Activation('sigmoid')(x)

    ''' Get model '''
    return models.Model(inp, outp), hparams

''' Model: Embedding 3CNN+BiLSTM '''
def model_lstmEm(
    inp_shape=[1000, 4], 
    embed_n=4,
    embed_dim=64,
    kernel=9,
    filters=64,
    rnn_units=40):

    params = locals()

    inp = Input(shape=inp_shape)
    ''' Embedding layer or not '''
    if embed_dim > 0:
        x = layers.Lambda(tokenize, arguments={'n': embed_n, 'padding': 'valid'})(inp)
        x = layers.Embedding(4**embed_n, embed_dim, input_length=[1000 - embed_n + 1])(x)
    else:
        if embed_n == 1:
            x = inp
        else:
            x = layers.Lambda(tokenize, arguments={'n': embed_n, 'padding': 'valid'})(inp)
            x = layers.Lambda(one_hot_layer, arguments={'n': 4**embed_n})(x)

    x = layers.Conv1D(filters, kernel)(inp)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.MaxPooling1D()(x)

    x = layers.Conv1D(filters, kernel)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPooling1D()(x)
    
    x = layers.Conv1D(filters, kernel)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPooling1D()(x)

    x = layers.Bidirectional(layers.LSTM(
        rnn_units, 
        return_sequences=True, 
        dropout=0.5))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    x = layers.Activation('sigmoid')(x)
    
    return models.Model(inp, x), params

def get_model(name='cnn', hparams={}):
    if name == '3cnn':
        return model_cnn(**hparams)
    elif name == '3cnn_lstm':
        return three_CNN_LSTM(**hparams)
    elif name == '3cnn_lstmEm':
        return model_lstmEm(**hparams)
    else:
        raise Exception('Unsupport model:', name)

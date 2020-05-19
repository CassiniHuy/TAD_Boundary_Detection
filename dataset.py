#!/usr/bin/env python
# coding: utf-8

# Author: SCU Wuyuzhang College, Wei Chengan & Wan Ziyi
# Data: 2020/5

import h5py
import random
import numpy as np
import tensorflow as tf
from random import shuffle, randint
from sklearn.model_selection import train_test_split

''' Convert one-hot to n-gram(tensorflow tensor) '''
def ngram(X, win, padding='SAME'):
    W = tf.constant([[[4**k]] for k in range(win)])
    Y = tf.argmax(X, axis=-1)
    if win == 1:
        return Y
    Y = tf.cast(Y, dtype=tf.float32)
    W = tf.cast(W, dtype=tf.float32)
    Y = tf.expand_dims(Y, axis=-1)
    Y = tf.nn.conv1d(value=Y, filters=W, stride=1, padding=padding)
    Y = tf.cast(Y, dtype=tf.int64)
    Y = tf.squeeze(Y)
    return Y

''' Convert to win-gram data using tensorflow '''
def tokenize(datapath, win, save_path=None):
    ''' data: dm3.kc167.example.h5 '''

    ''' Retrieve data '''
    f = h5py.File(datapath, 'r')
    x_test, y_test = np.array(f["x_test"]), np.array(f["y_test"])
    x_train, y_train = np.array(f["x_train"]), np.array(f["y_train"])
    x_val, y_val = np.array(f["x_val"]), np.array(f["y_val"])
    f.close()

    ''' Conversion '''
    with tf.Session() as sess:
        X = tf.placeholder(shape=x_train.shape, dtype=tf.int64)
        Y = ngram(X, win, padding='VALID')
        x_train_ngram = Y.eval(session=sess, feed_dict={X:x_train})

        X = tf.placeholder(shape=x_val.shape, dtype=tf.int64)
        Y = ngram(X, win, padding='VALID')
        x_val_ngram = Y.eval(session=sess, feed_dict={X:x_val})

        X = tf.placeholder(shape=x_test.shape, dtype=tf.int64)
        Y = ngram(X, win, padding='VALID')
        x_test_ngram = Y.eval(session=sess, feed_dict={X:x_test})

    ''' Save data '''
    if save_path:
        nf = h5py.File(save_path, 'w')
        nf.create_dataset('x_train', data=x_train_ngram)
        nf.create_dataset('x_val', data=x_val_ngram)
        nf.create_dataset('x_test', data=x_test_ngram)
        nf.create_dataset('y_train', data=y_train)
        nf.create_dataset('y_val', data=y_val)
        nf.create_dataset('y_test', data=y_test)
        nf.close()

    return {
        'x_train':x_train_ngram, 
        'x_val':x_val_ngram, 
        'x_test':x_test_ngram, 
        'y_train':y_train, 
        'y_val':y_val, 
        'y_test':y_test
        }

def load_h5(datapath, datasets):
    ''' Retrieve data '''
    f = h5py.File(datapath, 'r')
    data = []
    for dataset in datasets:
        data.append(np.array(f[dataset]))
    return tuple(data)

def dataset_split(X_pos, Y_pos, X_neg, Y_neg, train_size=0.8, state=1):
    if train_size == 1:
        X = np.concatenate([X_pos, X_neg], axis=0)
        Y = np.concatenate([Y_pos, Y_neg], axis=0)
        return X, Y

    X_pos_train, X_pos_test, Y_pos_train, Y_pos_test = train_test_split(
        X_pos, Y_pos, test_size=1-train_size, random_state=state)
    X_neg_train, X_neg_test, Y_neg_train, Y_neg_test = train_test_split(
        X_neg, Y_neg, test_size=1-train_size, random_state=state)

    X_train = np.concatenate([X_pos_train, X_neg_train], axis=0)
    Y_train = np.concatenate([Y_pos_train, Y_neg_train], axis=0)
    X_test = np.concatenate([X_pos_test, X_neg_test], axis=0)
    Y_test = np.concatenate([Y_pos_test, Y_neg_test], axis=0)

    return X_train, Y_train, X_test, Y_test

def load_data(datapath, train_size=0.8, state=1):
    ''' Retrieve data '''
    Y_pos, X_pos, Y_neg, X_neg = load_h5(datapath, ["Y_pos", "X_pos", "Y_neg", "X_neg"])

    return dataset_split(X_pos, Y_pos, X_neg, Y_neg, train_size=train_size, state=state)


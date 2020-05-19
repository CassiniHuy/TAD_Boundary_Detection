#!/usr/bin/env python
# coding: utf-8

# Author: SCU Wuyuzhang College, Wei Chengan & Wan Ziyi
# Data: 2020/5

import os
import h5py
import json
import datetime
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras import regularizers
from tensorflow.keras import models
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

from cnn_models import get_model
from dataset import load_data, get_article_aug_data, load_gen_data

''' Set GPU settings '''
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

''' Logging'''
def log_to_json(log, path):
    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, regularizers.Regularizer):
                return str(obj)
            else:
                return super(MyEncoder, self).default(obj)

    with open(path, 'w') as hf:
        json.dump(log, hf, cls=MyEncoder)

def get_metrics(y_score, y_true):
    r = np.array([1 if i > 0.5 else 0 for i in y_score])
    tp = np.where(r[np.where(r == 1)] == y_true[np.where(r == 1)])[0].shape[0]
    fp = np.where(r[np.where(r == 1)] != y_true[np.where(r == 1)])[0].shape[0]
    tn = np.where(r[np.where(r == 0)] == y_true[np.where(r == 0)])[0].shape[0]
    fn = np.where(r[np.where(r == 0)] != y_true[np.where(r == 0)])[0].shape[0]
    def devide(up, down):
        try:
            result = up / down
        except ZeroDivisionError:
            result = -1
        return result
    accuracy = devide(tp + tn, tp + fp + tn + fn)
    sensitivity = devide(tp, tp + fn)
    specificity = devide(tn, tn + fp)
    precision = devide(tp, tp + fp)
    f1score = devide(2 * tp, 2 * tp + fp + fn)
    auc = metrics.roc_auc_score(y_true, y_score)
    mets = {
        'accuracy' : accuracy, 
        'sensitivity': sensitivity, 
        'specificity': specificity, 
        'precision': precision,
        'f1score': f1score,
        'auc': auc,
        'confusion_matrix': [[tp, fn], [fp, tn]]
        }
    return mets

''' Compute metrics of kfold training '''
def summary_kfold(eval_dir):
    dicts = []
    for fname in os.listdir(eval_dir):
        path = os.path.join(eval_dir, fname)
        with open(path, 'r') as f:
            ev = json.load(f)
            del ev['confusion_matrix']
            del ev['y_score']
            dicts.append(ev)
    summary = {
        'accuracy' : [], 
        'sensitivity': [], 
        'specificity': [], 
        'precision': [],
        'f1score': [],
        'auc': [],
        'loss': []
        }
    for d in dicts:
        for k, v in d.items():
            summary[k].append(v)
    mNs = {}
    for k in summary.keys():
        valid_list = list(filter(lambda x: False if x == -1 else True, summary[k]))
        mNs[k + '_mean'] = np.mean(valid_list)
        mNs[k + '_std'] = np.std(valid_list)
        mNs[k + '_outlier'] = len(summary[k]) - len(valid_list)
    for k, v in mNs.items():
        summary[k] = v
    return summary

''' Evaluate model '''
def evaluate_model(model, x, y, batch_size=30):
    test_hist = model.evaluate(x=x, y=y)
    predict_p = model.predict(x)
    p = np.array(predict_p).squeeze()
    eval_dict = get_metrics(p, y)
    eval_dict['loss'] = test_hist[0]
    eval_dict['y_score'] = p
    return eval_dict

def compile_model(model_name, hp, op='adam'):
    ''' Construct model '''
    model, hparams = get_model(name=model_name, hparams=hp)
    model.compile(loss='binary_crossentropy', optimizer=op, metrics=['acc'])
    # print(model.summary())
    return model, hparams

def train_kfold(
    log_dir, 
    hparams, 
    model_name, 
    k, 
    state, 
    X, 
    Y, 
    X_test=None, 
    Y_test=None, 
    X_train_add=None,
    Y_train_add=None,
    batch_size=20, 
    epochs=200):
    '''
    X: The training set
    X_test: The testing set
    X_train_add: The additional set for model training, used for the 2nd experiment in paper.
    '''
    ''' Training '''
    # Log
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    model_dir = os.path.join(log_dir, timestamp, 'models')
    os.makedirs(model_dir)
    hist_dir = os.path.join(log_dir, timestamp, 'history')
    os.makedirs(hist_dir)
    eval_dir = os.path.join(log_dir, timestamp, 'evaluate')
    os.makedirs(eval_dir)
    params_savename = os.path.join(log_dir, timestamp, 'params.json')
    summary_savename = os.path.join(log_dir, timestamp, 'summary.json')
    test_hist_savename = os.path.join(log_dir, timestamp, 'test.json')

    # Start training
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=state)
    fold = 1
    accs = []
    for train_index, val_index in kfold.split(X, Y):
        model, params = compile_model(model_name, hparams)
        if fold == 1:
            print(model.summary())
            print(params)
        print('\n' + '='*60 + ' Fold: ' + str(fold) + ' ' + '='*60 + '\n')
        # Callback functions
        model_savename = os.path.join(model_dir, 'model{0}.h5'.format(str(fold)))
        hist_savename = os.path.join(hist_dir, 'history{0}.json'.format(str(fold)))
        val_savename = os.path.join(eval_dir, 'evaluate{0}.json'.format(str(fold)))
        cb_list = [
            callbacks.ModelCheckpoint(
                filepath=model_savename,
                monitor='val_acc',
                save_best_only=True
            ),
            callbacks.EarlyStopping(
                monitor='acc',
                patience=6,
            )
        ]
        # Add new training sets
        try:
            if X_train_add.any() and Y_train_add.any():
                x_train = np.concatenate([X[train_index], X_train_add], axis=0)
                y_train = np.concatenate([Y[train_index], Y_train_add], axis=0)
                index = list(range(len(y_train)))
                random.seed(state + 1)
                random.shuffle(index)
                x_train = x_train[index]
                y_train = y_train[index]
        except AttributeError:
            x_train = X[train_index]
            y_train = Y[train_index]

        history = model.fit(
            x_train,
            y_train,
            validation_data=(X[val_index], Y[val_index]),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=cb_list,
            verbose=2
        )
        # Log
        hist_dict = history.history
        m = models.load_model(model_savename, custom_objects={'tf': tf})
        val_dict = evaluate_model(m, X[val_index], Y[val_index])

        accs.append(val_dict['accuracy'])
        log_to_json(hist_dict, hist_savename)
        log_to_json(val_dict, val_savename)

        fold += 1
        K.clear_session()
        print('Session cleared.')
    # Summary
    try:
        if X_test.any() and Y_test.any():
            model_path = os.path.join(model_dir, 'model{0}.h5'.format(accs.index(max(accs))+1))
            m = models.load_model(model_path, custom_objects={'tf': tf})
            test_dict = evaluate_model(m, X_test, Y_test)
            log_to_json(test_dict, test_hist_savename)
    except AttributeError:
        pass
        
    log_to_json(hparams, params_savename)
    summary = summary_kfold(eval_dir)
    print(summary)
    log_to_json(summary, summary_savename)

if __name__ == '__main__':

    datapath = r'dataset\seqs_dm3\dm3_seqs.h5'
    
    hparams = {
        'inp_shape': [1000, 4], 
        'embed_n': 4,
        'embed_dim': 64, 
        'cnn_filters': [32], 
        'cnn_kernels': [2],
        'cnn_dilations': [[1, 1, 1]],
        'cnn_dropouts': [0.5],
        'cnn_regularizers': None, 
        'pooling': 'local',
        'max_pool': 2,
        'dense_regularizer': None,
        'batchnormal': False
    }

    X_train, Y_train, X_test, Y_test = load_data(datapath, train_size=0.8, state=99)

    train_kfold(
        log_dir=r'test', 
        hparams=hparams, model_name='cnn', k=10, state=1, 
        X=X_train, Y=Y_train, X_test=X_test, Y_test=Y_test, X_train_add=None, Y_train_add=None,
        batch_size=50, epochs=200)


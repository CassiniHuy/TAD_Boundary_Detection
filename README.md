# TAD Boundary Detection

## Requirements

Following packages are required, and python3 certainly.

1. tensorflow >= 1.14.0
2. numpy >= 1.16.2
3. sklearn >= 0.21.3

## Training

Train a model by a configuration dictionary.

For example, train a "3CNN(4,3)" model, *embed_n* for $H$(sliding window size). :
```
    datapath = r'dataset/dm3_seqs.h5'
    hparams = {
        'inp_shape': [1000, 4], 
        'embed_n': 4,
        'embed_dim': 64, 
        'cnn_filters': [32], 
        'cnn_kernels': [3],
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
        log_dir=r'test', hparams=hparams, model_name='3cnn', k=10, state=1, X=X_train, Y=Y_train, X_test=X_test, Y_test=Y_test, X_train_add=None, Y_train_add=None, batch_size=50, epochs=200)
```
When *embed_dim* is assigned with a negative value, one-hot encoding method is used, instead of an embedding layer.
 Train a "3CNN" model:
```
    hparams = {
        'inp_shape': [1000, 4], 
        'embed_n': 1,
        'embed_dim': -1, 
        'cnn_filters': [64], 
        'cnn_kernels': [9],
        'cnn_dilations': [[1, 1, 1]],
        'cnn_dropouts': [0.5],
        'cnn_regularizers': None, 
        'pooling': 'local',
        'max_pool': 2,
        'dense_regularizer': None,
        'batchnormal': False
    }
```

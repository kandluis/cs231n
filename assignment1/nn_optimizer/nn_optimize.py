from __future__ import print_function

import sys
sys.path.append('../')

from cs231n.classifiers.neural_net import TwoLayerNet
from cs231n.data_utils import load_CIFAR10

import numpy as np

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = '../cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test

# Write a function like this called 'main'
def main(job_id, params):
    print('Training new neural net and calculating validation value job #%d' % job_id)
    np.random.seed(0)  # For repeatability.
    X_train, y_train, X_val, y_val, _, _ = get_CIFAR10_data()
    
    input_size = 32 * 32 * 3
    num_classes = 10
    hidden_size = params['hidden_size']
    lr = params['learning_rate']
    lrdc = params['learning_rate_decay']
    r = params['regularization']
    net = TwoLayerNet(input_size, hidden_size, num_classes) 
    stats = net.train(X_train, y_train, X_val, y_val,
                      num_iters=600, batch_size=200,
                      learning_rate=lr, learning_rate_decay=lrdc,
                      reg=r)

    val_acc = (net.predict(X_val) == y_val).mean()
    print('Validation accuracy: ', val_acc)
    
    # We wish to maximize the validation accuracy, so minimize the negative.
    return -val_acc


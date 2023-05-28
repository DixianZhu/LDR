import tensorflow as tf
import numpy as np


IMG_SIZE = 32


def fMNIST():
    (train_data, train_label), (test_data, test_label) = tf.keras.datasets.fashion_mnist.load_data()
    (train_data, train_label) = (np.expand_dims(train_data.astype(float), axis=1), train_label.astype(np.int32))
    (test_data, test_label) = (np.expand_dims(test_data.astype(float), axis=1), test_label.astype(np.int32)) 
    print(train_data.shape) 
    print(test_data.shape) 
    return (train_data, train_label), (test_data, test_label)

def MNIST():
    (train_data, train_label), (test_data, test_label) = tf.keras.datasets.mnist.load_data()
    (train_data, train_label) = (np.expand_dims(train_data.astype(float), axis=1), train_label.astype(np.int32))
    (test_data, test_label) = (np.expand_dims(test_data.astype(float), axis=1), test_label.astype(np.int32)) 
    print(train_data.shape) 
    print(test_data.shape) 
    return (train_data, train_label), (test_data, test_label)


def CIFAR10():
    (train_data, train_label), (test_data, test_label) = tf.keras.datasets.cifar10.load_data()
    (train_data, train_label) = (train_data.astype(float), train_label.astype(np.int32))
    (test_data, test_label) = (test_data.astype(float), test_label.astype(np.int32)) 
    print(train_data.shape) 
    print(test_data.shape) 
    return (train_data, train_label), (test_data, test_label)


def CIFAR100():
    (train_data, train_label), (test_data, test_label) = tf.keras.datasets.cifar100.load_data()
    (train_data, train_label) = (train_data.astype(float), train_label.astype(np.int32))
    (test_data, test_label) = (test_data.astype(float), test_label.astype(np.int32)) 
    return (train_data, train_label), (test_data, test_label)



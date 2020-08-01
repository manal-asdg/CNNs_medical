# coding=utf-8
from __future__ import print_function
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import pickle
import json
import time
import my_parser as ps
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import input_data_2
import keras
import h5py
print("Loading unlabeled data......")
with open('unlabelled_new', 'rb') as fo:
    unlabels = pickle.load(fo, encoding='bytes')
X_unlabel, Y_unlabel = ps.parseUnlabel(unlabels, 2, 'rgb')
N_unlabel = unlabels['names']
# print(N_unlabel)
with h5py.File('unlabelled_new.h5', 'w') as f:
    f['names'] = [np.string_(i) for i in N_unlabel]
    f['data'] = X_unlabel
    f['labels'] = Y_unlabel

# f = h5py.File("label+_data.h5", "r")
# X_trainplus = f['data'][:]
# Y_trainplus = f['labels'][:]
# f.close()
#
# print(Y_trainplus.size)

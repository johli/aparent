from __future__ import print_function
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, LSTM, ConvLSTM2D, BatchNormalization
from keras.layers import Concatenate, Reshape
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
from keras import backend as K
import keras.losses

import tensorflow as tf

import pandas as pd

import os
import pickle
import numpy as np

import scipy.sparse as sp
import scipy.io as spio

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import isolearn_keras as iso

from aparent_losses import *

from aparent_data_plasmid_no_pseudo import *
from aparent_model_plasmid_large_lessdropout_batchnorm import *

#Trainer parameters
load_saved_model = False

#Trained model suffixes:
#all_libs
#all_libs_no_sampleweights
#all_libs_highcount
#all_libs_highcount_no_sampleweights

#all_libs_no_sampleweights_partitioned
#all_libs_no_sampleweights_partitioned_adam


load_name_suffix = 'all_libs_no_sampleweights_adam_retry_1'
save_name_suffix = 'all_libs_no_sampleweights_adam_retry_1'
epochs = 10#15
batch_size = 32

#Filter sublibraries
kept_libraries = None
#kept_libraries = [2, 5, 8, 11, 20, 22, 30, 31, 34]

#Load plasmid data #_w_array_part_1
data_gens = load_data(batch_size=batch_size, valid_set_size=0.025, test_set_size=0.025, data_version='', kept_libraries=kept_libraries)

#Load model definition
models = load_aparent_model(batch_size, use_sample_weights=False)
_, loss_model = models[-1]


#Optimizer code
save_dir = os.path.join(os.getcwd(), 'saved_models')

checkpoint_dir = os.path.join(os.getcwd(), 'model_checkpoints')
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

if load_saved_model :
    for model_name_prefix, model in models[:-1] :
        model_name = 'aparent_' + model_name_prefix + '_' + load_name_suffix + '.h5'
        model_path = os.path.join(save_dir, model_name)
        saved_model = load_model(model_path)
        
        model.set_weights(saved_model.get_weights())

#opt = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

loss_model.compile(loss=lambda true, pred: pred, optimizer=opt)

callbacks =[
    ModelCheckpoint(os.path.join(checkpoint_dir, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1),
    EarlyStopping(monitor='val_loss', min_delta=0.0002, patience=3, verbose=0, mode='auto')
]

loss_model.fit_generator(generator=data_gens['train'],
                    validation_data=data_gens['valid'],
                    epochs=epochs,
                    use_multiprocessing=True,
                    workers=4,
                    callbacks=callbacks)


# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

for model_name_prefix, model in models[:-1] :
    model_name = 'aparent_' + model_name_prefix + '_' + save_name_suffix + '.h5'
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

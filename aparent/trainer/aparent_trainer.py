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

import isolearn.keras as iso

from aparent.losses import *

from aparent.data.aparent_data_plasmid import load_data
from aparent.model.aparent_model_plasmid_large_lessdropout import load_aparent_model


#Wrapper function to execute APARENT trainer
def run_trainer(load_data_func, load_model_func, load_saved_model, save_dir_path, load_name_suffix, save_name_suffix, epochs, batch_size, use_sample_weights, valid_set_size, test_set_size, kept_libraries) :

    #Load plasmid data #_w_array_part_1
    data_gens = load_data_func(batch_size=batch_size, valid_set_size=valid_set_size, test_set_size=test_set_size, data_version='', kept_libraries=kept_libraries)

    #Load model definition
    models = load_model_func(batch_size, use_sample_weights=use_sample_weights)
    _, loss_model = models[-1]


    #Optimizer code
    save_dir = os.path.join(os.getcwd(), save_dir_path)

    checkpoint_dir = os.path.join(os.getcwd(), 'model_checkpoints')
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if load_saved_model :
        for model_name_prefix, model in models[:-1] :
            model_name = 'aparent_' + model_name_prefix + '_' + load_name_suffix + '.h5'
            model_path = os.path.join(save_dir, model_name)
            saved_model = load_model(model_path)
            
            model.set_weights(saved_model.get_weights())

    opt = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

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


#Execute Trainer if called from cmd-line
if __name__ == "__main__" :

    #Trainer parameters
    load_saved_model = False
    save_dir_path = '../../saved_models'
    load_name_suffix = 'all_libs_no_sampleweights'
    save_name_suffix = 'all_libs_no_sampleweights'
    epochs = 15
    batch_size = 32

    use_sample_weights = False

    valid_set_size = 0.025
    test_set_size = 0.025

    #Filter sublibraries
    kept_libraries = None
    #kept_libraries = [2, 5, 8, 11, 20, 22, 30, 31, 34]

    run_trainer(load_data, load_aparent_model, load_saved_model, save_dir_path, load_name_suffix, save_name_suffix, epochs, batch_size, use_sample_weights, valid_set_size, test_set_size, kept_libraries)

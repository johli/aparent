from __future__ import print_function
import keras
from keras import backend as K

import tensorflow as tf

import pandas as pd

import os
import pickle
import numpy as np

import scipy.sparse as sp
import scipy.io as spio

import isolearn.io as isoio
import isolearn.keras as iso

def iso_normalizer(t) :
    iso = 0.0
    if np.sum(t) > 0.0 :
        iso = np.sum(t[77: 77+30]) / np.sum(t)
    
    return iso

def cut_normalizer(t) :
    cuts = np.concatenate([np.zeros(205), np.array([1.0])])
    if np.sum(t) > 0.0 :
        cuts = t / np.sum(t)
    
    return cuts

def load_data(batch_size=32, valid_set_size=0.0, test_set_size=1.0, file_path='', data_version='') :

    #Load array data
    array_dict = isoio.load(file_path + 'apa_array_data' + data_version)
    array_df = array_dict['array_df']
    array_cuts = array_dict['pooled_cuts']

    array_index = np.arange(len(array_df), dtype=np.int)

    print('Designed MPRA size = ' + str(array_index.shape[0]))
    
    #Generate training and test set indexes
    array_index = np.arange(len(array_df), dtype=np.int)

    array_train_index = array_index[:-int(len(array_df) * (valid_set_size + test_set_size))]
    array_valid_index = array_index[array_train_index.shape[0]:-int(len(array_df) * test_set_size)]
    array_test_index = array_index[array_train_index.shape[0] + array_valid_index.shape[0]:]

    print('Training set size = ' + str(array_train_index.shape[0]))
    print('Validation set size = ' + str(array_valid_index.shape[0]))
    print('Test set size = ' + str(array_test_index.shape[0]))

    unique_libraries = np.array(['tomm5_up_n20c20_dn_c20', 'tomm5_up_c20n20_dn_c20', 'tomm5_up_n20c20_dn_n20', 'tomm5_up_c20n20_dn_n20', 'doubledope', 'simple', 'atr', 'hsp', 'snh', 'sox', 'wha', 'array', 'aar'], dtype=np.object)

    array_gens = {
        gen_id : iso.DataGenerator(
            idx,
            {'df' : array_df, 'cuts' : array_cuts},
            batch_size=batch_size,
            inputs = [
                {
                    'id' : 'seq',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : iso.SequenceExtractor('seq_ext', start_pos=180, end_pos=180 + 205),
                    'encoder' : iso.OneHotEncoder(seq_length=205),
                    'dim' : (205, 4, 1),
                    'sparsify' : False
                },
                {
                    'id' : 'lib',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: 'array',
                    'encoder' : iso.CategoricalEncoder(n_categories=len(unique_libraries), categories=unique_libraries),
                    'sparsify' : False
                },
                {
                    'id' : 'distal_pas',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: 1,
                    'encoder' : None,
                    'sparsify' : False
                }
            ],
            outputs = [
                {
                    'id' : 'prox_usage',
                    'source_type' : 'matrix',
                    'source' : 'cuts',
                    'extractor' : iso.CountExtractor(start_pos=180, end_pos=180 + 205, static_poses=[-1], sparse_source=True),
                    'transformer' : lambda t: iso_normalizer(t),
                    'sparsify' : False
                }
            ],
            randomizers = [],
            shuffle = False
        ) for gen_id, idx in [('all', array_index), ('train', array_train_index), ('valid', array_valid_index), ('test', array_test_index)]
    }

    return array_gens

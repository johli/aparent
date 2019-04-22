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

def load_data(batch_size=32, valid_set_size=0.0, test_set_size=1.0, file_path='') :

    #Load array data
    array_dict = isoio.load(file_path + 'apa_array_data_master_seq')
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

    
    #Manually set sublibrary intercept terms
    array_df['library_index'] = np.zeros(len(array_df), dtype=np.int)
    array_df['distal_pas'] = np.ones(len(array_df))
    array_df.loc[array_df['gene'] == 'doubledope', 'library_index'] = 20
    array_df.loc[array_df['gene'] == 'doubledope', 'distal_pas'] = 1
    array_df.loc[array_df['gene'] == 'simple', 'library_index'] = 22
    array_df.loc[array_df['gene'] == 'simple', 'distal_pas'] = 0
    array_df.loc[array_df['gene'] == 'tomm5', 'library_index'] = 8
    array_df.loc[array_df['gene'] == 'tomm5', 'distal_pas'] = 1
    array_df.loc[array_df['gene'] == 'aar', 'library_index'] = 30
    array_df.loc[array_df['gene'] == 'aar', 'distal_pas'] = 0
    array_df.loc[array_df['gene'] == 'atr', 'library_index'] = 31
    array_df.loc[array_df['gene'] == 'atr', 'distal_pas'] = 0
    array_df.loc[array_df['gene'] == 'hsp', 'library_index'] = 32
    array_df.loc[array_df['gene'] == 'hsp', 'distal_pas'] = 0
    array_df.loc[array_df['gene'] == 'snh', 'library_index'] = 33
    array_df.loc[array_df['gene'] == 'snh', 'distal_pas'] = 0
    array_df.loc[array_df['gene'] == 'sox', 'library_index'] = 34
    array_df.loc[array_df['gene'] == 'sox', 'distal_pas'] = 0
    array_df.loc[array_df['gene'] == 'wha', 'library_index'] = 35
    array_df.loc[array_df['gene'] == 'wha', 'distal_pas'] = 0


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
                    'extractor' : iso.SequenceExtractor('seq_ext', start_pos=200 + 1, end_pos=200 + 1 + 185),
                    'encoder' : iso.OneHotEncoder(seq_length=185),
                    'dim' : (1, 185, 4),
                    'sparsify' : False
                },
                {
                    'id' : 'lib',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: row['library_index'],
                    'encoder' : iso.CategoricalEncoder(n_categories=36, categories=np.arange(36, dtype=np.int).tolist()),
                    'sparsify' : False
                },
                {
                    'id' : 'distal_pas',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: row['distal_pas'],
                    'encoder' : None,
                    'sparsify' : False
                }
            ],
            outputs = [
                {
                    'id' : 'prox_usage',
                    'source_type' : 'matrix',
                    'source' : 'cuts',
                    'extractor' : iso.CountExtractor(start_pos=200 + 1, end_pos=200 + 1 + 185, static_poses=[-1], sparse_source=True),
                    'transformer' : lambda t: iso_normalizer(t),
                    'sparsify' : False
                }
            ],
            randomizers = [],
            shuffle = False
        ) for gen_id, idx in [('all', array_index), ('train', array_train_index), ('valid', array_valid_index), ('test', array_test_index)]
    }

    return array_gens

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


def load_data(batch_size=32, valid_set_size=0.0, test_set_size=1.0, file_path='') :

    #Load array data
    native_dict = isoio.load(file_path + 'apa_leslie_apadb_data')
    native_df = native_dict['df']

    native_index = np.arange(len(native_df), dtype=np.int)

    print('Native pA (APADB + Leslie) size = ' + str(native_index.shape[0]))

    native_train_index = native_index[:-int(len(native_df) * (valid_set_size + test_set_size))]
    native_valid_index = native_index[native_train_index.shape[0]:-int(len(native_df) * test_set_size)]
    native_test_index = native_index[native_train_index.shape[0] + native_valid_index.shape[0]:]

    print('Training set size = ' + str(native_train_index.shape[0]))
    print('Validation set size = ' + str(native_valid_index.shape[0]))
    print('Test set size = ' + str(native_test_index.shape[0]))

    native_gens = {
        gen_id : iso.DataGenerator(
            idx,
            {'df' : native_df},
            batch_size=batch_size,
            inputs = [
                {
                    'id' : 'seq_prox',
                    'source' : 'df',
                    'source_type' : 'dataframe',
                    'extractor' : iso.SequenceExtractor('wide_seq_ext', start_pos=105, end_pos=105 + 205),
                    'encoder' : iso.OneHotEncoder(seq_length=205),
                    'dim' : (205, 4, 1),
                    'sparsify' : False
                },
                {
                    'id' : 'lib',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: np.zeros(13),
                    'encoder' : None,
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
                    'id' : 'dummy_output',
                    'source_type' : 'zeros',
                    'dim' : (1,),
                    'sparsify' : False
                }
            ],
            randomizers = [],
            shuffle = False
        ) for gen_id, idx in [('all', native_index), ('train', native_train_index), ('valid', native_valid_index), ('test', native_test_index)]
    }

    return native_gens

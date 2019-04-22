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


def load_data(batch_size=32, valid_set_size=0.025, test_set_size=0.025, file_path='', kept_libraries=None) :

    #Load plasmid data
    #plasmid_dict = pickle.load(open('apa_plasmid_data' + data_version + '.pickle', 'rb'))
    plasmid_dict = isoio.load(file_path + 'apa_plasmid_data_legacy')
    plasmid_df = plasmid_dict['plasmid_df']
    plasmid_cuts = plasmid_dict['plasmid_cuts']
    
    if kept_libraries is not None :
        keep_index = np.nonzero(plasmid_df.library_index.isin(kept_libraries))[0]
        plasmid_df = plasmid_df.iloc[keep_index].copy()
        plasmid_cuts = plasmid_cuts[keep_index, :]
    
    #Generate training and test set indexes
    plasmid_index = np.arange(len(plasmid_df), dtype=np.int)

    plasmid_train_index = plasmid_index[:-int(len(plasmid_df) * (valid_set_size + test_set_size))]
    plasmid_valid_index = plasmid_index[plasmid_train_index.shape[0]:-int(len(plasmid_df) * test_set_size)]
    plasmid_test_index = plasmid_index[plasmid_train_index.shape[0] + plasmid_valid_index.shape[0]:]

    print('Training set size = ' + str(plasmid_train_index.shape[0]))
    print('Validation set size = ' + str(plasmid_valid_index.shape[0]))
    print('Test set size = ' + str(plasmid_test_index.shape[0]))
    
    

    plasmid_prediction_gens = {
        gen_id : iso.DataGenerator(
            idx,
            {'df' : plasmid_df, 'cuts' : plasmid_cuts},
            batch_size=batch_size,
            inputs = [
                {
                    'id' : 'seq',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : iso.SequenceExtractor('seq', start_pos=1, end_pos=1 + 185),
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
                    'extractor' : lambda row, index: 1 if row['library_index'] in [2, 5, 8, 11, 20] else 0,
                    'encoder' : None,
                    'sparsify' : False
                }
            ],
            outputs = [
                {
                    'id' : 'prox_usage',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: row['proximal_count'] / row['total_count'],
                    'transformer' : lambda t: t,
                    'dim' : (1,),
                    'sparsify' : False
                },
                {
                    'id' : 'prox_cuts',
                    'source_type' : 'matrix',
                    'source' : 'cuts',
                    'extractor' : iso.CountExtractor(start_pos=0, end_pos=186, sparse_source=False),
                    'transformer' : lambda t: t,
                    'dim' : (186,),
                    'sparsify' : False
                }
            ],
            randomizers = [],
            shuffle = False,
            densify_batch_matrices=True
        ) for gen_id, idx in [('all', plasmid_index), ('train', plasmid_train_index), ('valid', plasmid_valid_index), ('test', plasmid_test_index)]
    }

    return plasmid_prediction_gens

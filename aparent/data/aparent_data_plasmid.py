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

import isolearn_keras as iso

def iso_normalizer(t) :
    iso = 0.0
    if np.sum(t) > 0.0 :
        iso = np.sum(t[80: 80+25]) / np.sum(t)
    
    return iso

def cut_normalizer(t) :
    cuts = np.concatenate([np.zeros(205), np.array([1.0])])
    if np.sum(t) > 0.0 :
        cuts = t / np.sum(t)
    
    return cuts

def load_data(batch_size=32, valid_set_size=0.025, test_set_size=0.025, data_version='', kept_libraries=None) :

    #Load plasmid data
    plasmid_dict = pickle.load(open('apa_plasmid_data' + data_version + '.pickle', 'rb'))
    plasmid_df = plasmid_dict['plasmid_df']
    plasmid_cuts = plasmid_dict['plasmid_cuts']
    
    unique_libraries = plasmid_df['library'].unique()
    
    if kept_libraries is not None :
        keep_index = np.nonzero(plasmid_df.library_index.isin(kept_libraries))[0]
        plasmid_df = plasmid_df.iloc[keep_index].copy()
        plasmid_cuts = plasmid_cuts[keep_index, :]

    #Take only a subset of examples
    #plasmid_df = plasmid_df.iloc[-10000:].copy()
    #plasmid_cuts = plasmid_cuts[-10000:]
    
    #Generate training and test set indexes
    plasmid_index = np.arange(len(plasmid_df), dtype=np.int)

    plasmid_train_index = plasmid_index[:-int(len(plasmid_df) * (valid_set_size + test_set_size))]
    plasmid_valid_index = plasmid_index[plasmid_train_index.shape[0]:-int(len(plasmid_df) * valid_set_size)]
    plasmid_test_index = plasmid_index[plasmid_train_index.shape[0] + plasmid_valid_index.shape[0]:]

    print('Training set size = ' + str(plasmid_train_index.shape[0]))
    print('Validation set size = ' + str(plasmid_valid_index.shape[0]))
    print('Test set size = ' + str(plasmid_test_index.shape[0]))
    
    
    pos_shifter = iso.get_bellcurve_shifter()

    prox_range = (np.arange(25, dtype=np.int) + 80).tolist()
    norm_range = np.arange(206).tolist()

    plasmid_gens = {
        gen_id : iso.DataGenerator(
            idx,
            {'df' : plasmid_df, 'cuts' : plasmid_cuts},
            batch_size=batch_size,
            inputs = [
                {
                    'id' : 'seq',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : iso.SequenceExtractor('padded_seq', start_pos=180, end_pos=180 + 205, shifter=pos_shifter if gen_id == 'train' else None),
                    'encoder' : iso.OneHotEncoder(seq_length=205),
                    'dim' : (205, 4, 1),
                    'sparsify' : False
                },
                {
                    'id' : 'lib',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: row['library'],
                    'encoder' : iso.CategoricalEncoder(n_categories=len(unique_libraries), categories=unique_libraries),
                    'sparsify' : False
                },
                {
                    'id' : 'distal_pas',
                    'source_type' : 'dataframe',
                    'source' : 'df',
                    'extractor' : lambda row, index: 1 if row['library_index'] in [2, 5, 8, 11, 20, 40] else 0,
                    'encoder' : None,
                    'sparsify' : False
                },
                {
                    'id' : 'total_count',
                    'source_type' : 'matrix',
                    'source' : 'cuts',
                    'extractor' : iso.CountExtractor(start_pos=180, end_pos=180 + 205, static_poses=[-1], shifter=pos_shifter if gen_id == 'train' else None, sparse_source=False),
                    'transformer' : lambda t: np.sum(t),
                    'dim' : (1,),
                    'sparsify' : False
                },
                {
                    'id' : 'prox_usage',
                    'source_type' : 'matrix',
                    'source' : 'cuts',
                    'extractor' : iso.CountExtractor(start_pos=180, end_pos=180 + 205, static_poses=[-1], shifter=pos_shifter if gen_id == 'train' else None, sparse_source=False),
                    'transformer' : lambda t: iso_normalizer(t),
                    'dim' : (1,),
                    'sparsify' : False
                },
                {
                    'id' : 'prox_cuts',
                    'source_type' : 'matrix',
                    'source' : 'cuts',
                    'extractor' : iso.CountExtractor(start_pos=180, end_pos=180 + 205, static_poses=[-1], shifter=pos_shifter if gen_id == 'train' else None, sparse_source=False),
                    'transformer' : lambda t: cut_normalizer(t),
                    'dim' : (206,),
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
            randomizers = [pos_shifter] if gen_id == 'train' else [],
            shuffle = True,
            densify_batch_matrices=True
        ) for gen_id, idx in [('train', plasmid_train_index), ('valid', plasmid_valid_index), ('test', plasmid_test_index)]
    }

    return plasmid_gens

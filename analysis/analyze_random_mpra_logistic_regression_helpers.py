import pandas as pd

import os
import pickle
import numpy as np

import scipy.sparse as sp
import scipy.io as spio

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import isolearn.io as isoio
import isolearn.keras as iso

import scipy.optimize as spopt
from scipy.stats import pearsonr

#Logistic regression prediction
def get_y_pred(X, w, w_0) :
    return 1. / (1. + np.exp(-1. * (X.dot(w) + w_0)))

#Safe log for NLL
def safe_log(x, minval=0.01):
    return np.log(x.clip(min=minval))

#Logistic regression NLL loss
def log_loss(w_bundle, *fun_args) :
    (X, y, lambda_penalty) = fun_args
    w = w_bundle[1:]
    w_0 = w_bundle[0]
    N = float(X.shape[0])

    log_y_zero = safe_log(1. - get_y_pred(X, w, w_0))
    log_y_one = safe_log(get_y_pred(X, w, w_0))

    log_loss = (1. / 2.) * lambda_penalty * np.square(np.linalg.norm(w)) - (1. / N) * np.sum(y * log_y_one + (1. - y) * log_y_zero)

    return log_loss

#Logistic regression NLL gradient
def log_loss_gradient(w_bundle, *fun_args) :
    (X, y, lambda_penalty) = fun_args
    w = w_bundle[1:]
    w_0 = w_bundle[0]
    N = float(X.shape[0])

    y_pred = get_y_pred(X, w, w_0)

    w_0_gradient = - (1. / N) * np.sum(y - y_pred)
    w_gradient = 1. * lambda_penalty * w - (1. / N) * X.T.dot(y - y_pred)

    return np.concatenate([[w_0_gradient], w_gradient])

def mask_constant_sequence_regions(df) :
	mask_dict = {
	    2  : 'XXXXXNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
	    5  : 'XXXXXXXXXXXXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
	    8  : 'XXXXXNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
	    11 : 'XXXXXXXXXXXXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
	    20 : 'XXXXXXXXXXXXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNXXXXX',
	    22 : 'XXXXXNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
	    30 : 'XXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
	    31 : 'XXXXXXXXXNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
	    32 : 'XXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
	    33 : 'XXNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNNNNXNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
	    34 : 'XXXXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
	    35 : 'XXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXNNNNNNNNNNNNNNNNNNNNNNNNNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
	}

	def map_mask(row) :
	    if row['library_index'] not in mask_dict :
	        return ('X' * 186)
	    library_mask = mask_dict[row['library_index']]
	    seq_var = ''
	    seq = row['seq']
	    for j in range(0, len(seq)) :
	        if library_mask[j] == 'N' :
	            seq_var += seq[j]
	        else :
	            seq_var += 'X'
	    
	    return seq_var

	df['seq_var'] = df.apply(map_mask, axis=1)

	return df

def align_on_cse(df, align_dict={20 : True}) :
	pas_1 = 'AATAAA'
	pas_2 = 'ATTAAA'

	cano_pas1 = 'AATAAA'
	cano_pas2 = 'ATTAAA'

	pas_mutex1_1 = {}
	pas_mutex1_2 = {}

	pas_mutex2_1 = {}

	for pos in range(0, 6) :
	    for base in ['A', 'C', 'G', 'T'] :
	        if cano_pas1[:pos] + base + cano_pas1[pos+1:] not in pas_mutex1_1 :
	            pas_mutex1_1[cano_pas1[:pos] + base + cano_pas1[pos+1:]] = True
	        if cano_pas2[:pos] + base + cano_pas2[pos+1:] not in pas_mutex1_2 :
	            pas_mutex1_2[cano_pas2[:pos] + base + cano_pas2[pos+1:]] = True

	for pos1 in range(0, 6) :
	    for pos2 in range(pos1 + 1, 6) :
	        for base1 in ['A', 'C', 'G', 'T'] :
	            for base2 in ['A', 'C', 'G', 'T'] :
	                if cano_pas1[:pos1] + base1 + cano_pas1[pos1+1:pos2] + base2 + cano_pas1[pos2+1:] not in pas_mutex2_1 :
	                    pas_mutex2_1[cano_pas1[:pos1] + base1 + cano_pas1[pos1+1:pos2] + base2 + cano_pas1[pos2+1:]] = True

	def align_on_pas(row) :
	    align_up = 3
	    align_down = 3
	    
	    align_index = 50
	    new_align_index = 50
	    
	    align_score = 0
	    
	    if row['library_index'] not in align_dict :
	        return row['seq_var']
	    
	    for j in range(align_index - align_up, align_index + align_up) :
	        candidate_pas = row['seq'][j:j+6]
	        
	        if candidate_pas == cano_pas1 :
	            new_align_index = j
	            align_score = 4
	        elif candidate_pas == cano_pas2 and align_score < 3 :
	            new_align_index = j
	            align_score = 3
	        elif candidate_pas in pas_mutex1_1 and align_score < 2 :
	            new_align_index = j
	            align_score = 2
	        elif candidate_pas in pas_mutex2_1 and align_score < 1 :
	            new_align_index = j
	            align_score = 1
	    
	    seq_aligned = row['seq_var']
	    if align_score > 0 :
	        align_diff = int(new_align_index - align_index)
	        
	        if align_diff > 0 :
	            seq_aligned = seq_aligned[align_diff:] + ('X' * align_diff)
	        elif align_diff < 0 :
	            align_diff = np.abs(align_diff)
	            seq_aligned = ('X' * align_diff) + seq_aligned[:-align_diff]
	        
	        if len(seq_aligned) != 186 :
	            print('ERROR')
	            print(align_diff)
	            print(row['seq_var'])
	            print(seq_aligned)
	    
	    return seq_aligned

	df['seq_var_aligned'] = df.apply(align_on_pas, axis=1)

	return df


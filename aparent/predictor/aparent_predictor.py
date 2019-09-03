import keras
from keras.models import Sequential, Model, load_model
from keras import backend as K

import tensorflow as tf

import os

import numpy as np

from scipy.signal import convolve as sp_conv
from scipy.signal import correlate as sp_corr
from scipy.signal import find_peaks

class OneHotEncoder :
    def __init__(self, seq_length=100, default_fill_value=0) :
        self.seq_length = seq_length
        self.default_fill_value = default_fill_value
        self.encode_map = {
            'A' : 0,
            'C' : 1,
            'G' : 2,
            'T' : 3
        }
        self.decode_map = {
            0 : 'A',
            1 : 'C',
            2 : 'G',
            3 : 'T',
            -1 : 'X'
        }
    
    def encode(self, seq) :
        one_hot = np.zeros((self.seq_length, 4))
        self.encode_inplace(seq, one_hot)

        return one_hot
    
    def encode_inplace(self, seq, encoding) :
        for pos, nt in enumerate(list(seq)) :
            if nt in self.encode_map :
                encoding[pos, self.encode_map[nt]] = 1
            elif self.default_fill_value != 0 :
                encoding[pos, :] = self.default_fill_value
    
    def __call__(self, seq) :
        return self.encode(seq)

def logit(x) :
	return np.log(x / (1.0 - x))

def get_aparent_encoder(lib_bias=None) :
	onehot_encoder = OneHotEncoder(205)

	def encode_for_aparent(sequences) :
		one_hots = np.concatenate([np.reshape(onehot_encoder(sequence), (1, len(sequence), 4, 1)) for sequence in sequences], axis=0)
		
		fake_lib = np.zeros((len(sequences), 13))
		fake_d = np.ones((len(sequences), 1))

		if lib_bias is not None :
			fake_lib[:, lib_bias] = 1.

		return [
			one_hots,
			fake_lib,
			fake_d
		]

	return encode_for_aparent

def get_aparent_legacy_encoder(lib_bias=None) :
	onehot_encoder = OneHotEncoder(185)

	def encode_for_aparent(sequences) :
		one_hots = np.concatenate([np.reshape(onehot_encoder(sequence), (1, 1, len(sequence), 4)) for sequence in sequences], axis=0)
		
		fake_lib = np.zeros((len(sequences), 36))
		fake_d = np.ones((len(sequences), 1))
		
		if lib_bias is not None :
			fake_lib[:, lib_bias] = 1.

		return [
			one_hots,
			fake_lib,
			fake_d
		]

	return encode_for_aparent

def get_apadb_encoder() :
	onehot_encoder = OneHotEncoder(205)

	def encode_for_apadb(prox_sequences, dist_sequences, prox_cut_starts, prox_cut_ends, dist_cut_starts, dist_cut_ends, site_distances) :
		prox_one_hots = np.concatenate([np.reshape(onehot_encoder(sequence), (1, len(sequence), 4, 1)) for sequence in prox_sequences], axis=0)
		dist_one_hots = np.concatenate([np.reshape(onehot_encoder(sequence), (1, len(sequence), 4, 1)) for sequence in dist_sequences], axis=0)

		return [
			prox_one_hots,
			dist_one_hots,
			np.array(prox_cut_starts).reshape(-1, 1),
			np.array(prox_cut_ends).reshape(-1, 1),
			np.array(dist_cut_starts).reshape(-1, 1),
			np.array(dist_cut_ends).reshape(-1, 1),
			np.log(np.array(site_distances).reshape(-1, 1)),
			np.zeros((len(prox_sequences), 13)),
			np.ones((len(prox_sequences), 1))
		]

	return encode_for_apadb

def find_polya_peaks(aparent_model, aparent_encoder, seq, sequence_stride=10, conv_smoothing=True, peak_min_height=0.01, peak_min_distance=50, peak_prominence=(0.01, None)) :
	cut_pred_padded_slices = []
	cut_pred_padded_masks = []

	start_pos = 0
	end_pos = 205
	while True :

		seq_slice = ''
		effective_len = 0

		if end_pos <= len(seq) :
			seq_slice = seq[start_pos: end_pos]
			effective_len = 205
		else :
			seq_slice = (seq[start_pos:] + ('X' * 200))[:205]
			effective_len = len(seq[start_pos:])

		_, cut_pred = aparent_model.predict(x=aparent_encoder([seq_slice]))

		#print("Striding over subsequence [" + str(start_pos) + ", " + str(end_pos) + "] (Total length = " + str(len(seq)) + ")...")

		padded_slice = np.concatenate([
			np.zeros(start_pos),
			np.ravel(cut_pred)[:effective_len],
			np.zeros(len(seq) - start_pos - effective_len),
			np.array([np.ravel(cut_pred)[205]])
		], axis=0)

		padded_mask = np.concatenate([
			np.zeros(start_pos),
			np.ones(effective_len),
			np.zeros(len(seq) - start_pos - effective_len),
			np.ones(1)
		], axis=0)[:len(seq)+1]

		cut_pred_padded_slices.append(padded_slice.reshape(1, -1))
		cut_pred_padded_masks.append(padded_mask.reshape(1, -1))

		if end_pos >= len(seq) :
			break

		start_pos += sequence_stride
		end_pos += sequence_stride

	cut_slices = np.concatenate(cut_pred_padded_slices, axis=0)[:, :-1]
	cut_masks = np.concatenate(cut_pred_padded_masks, axis=0)[:, :-1]
	
	if conv_smoothing :
		smooth_filter = np.array([
			[0.005, 0.01, 0.025, 0.05, 0.085, 0.175, 0.3, 0.175, 0.085, 0.05, 0.025, 0.01, 0.005]
		])

		cut_slices = sp_corr(cut_slices, smooth_filter, mode='same')
	
	
	avg_cut_pred = np.sum(cut_slices, axis=0) / np.sum(cut_masks, axis=0)
	std_cut_pred = np.sqrt(np.sum((cut_slices - np.expand_dims(avg_cut_pred, axis=0))**2, axis=0) / np.sum(cut_masks, axis=0))

	peak_ixs, _ = find_peaks(avg_cut_pred, height=peak_min_height, distance=peak_min_distance, prominence=peak_prominence)
	
	return peak_ixs.tolist(), avg_cut_pred

def score_polya_peaks(aparent_model, aparent_encoder, seq, peak_ixs, sequence_stride=2, strided_agg_mode='max', iso_scoring_mode='both', score_unit='log') :
	peak_iso_scores = []

	iso_pred_dict = {}
	iso_pred_from_cuts_dict = {}

	for peak_ix in peak_ixs :

		iso_pred_dict[peak_ix] = []
		iso_pred_from_cuts_dict[peak_ix] = []

		if peak_ix > 75 and peak_ix < len(seq) - 150 :
			for j in range(0, 30, sequence_stride) :
				seq_slice = (('X' * 35) + seq + ('X' * 35))[peak_ix + 35 - 80 - j: peak_ix + 35 - 80 - j + 205]

				if len(seq_slice) != 205 :
					continue

				iso_pred, cut_pred = aparent_model.predict(x=aparent_encoder([seq_slice]))

				iso_pred_dict[peak_ix].append(iso_pred[0, 0])
				iso_pred_from_cuts_dict[peak_ix].append(np.sum(cut_pred[0, 77: 107]))

		if len(iso_pred_dict[peak_ix]) > 0 :
			iso_pred = np.mean(iso_pred_dict[peak_ix])
			iso_pred_from_cuts = np.mean(iso_pred_from_cuts_dict[peak_ix])
			if strided_agg_mode == 'max' :
				iso_pred = np.max(iso_pred_dict[peak_ix])
				iso_pred_from_cuts = np.max(iso_pred_from_cuts_dict[peak_ix])
			elif strided_agg_mode == 'median' :
				iso_pred = np.median(iso_pred_dict[peak_ix])
				iso_pred_from_cuts = np.median(iso_pred_from_cuts_dict[peak_ix])

			if iso_scoring_mode == 'both' :
				peak_iso_scores.append((iso_pred + iso_pred_from_cuts) / 2.)
			elif iso_scoring_mode == 'from_iso' :
				peak_iso_scores.append(iso_pred)
			elif iso_scoring_mode == 'from_cuts' :
				peak_iso_scores.append(iso_pred_from_cuts)

			if score_unit == 'log' :
				peak_iso_scores[-1] = np.log(peak_iso_scores[-1] / (1. - peak_iso_scores[-1]))


			peak_iso_scores[-1] = round(peak_iso_scores[-1], 3)
		else :
			peak_iso_scores.append(-10)
	
	return peak_iso_scores


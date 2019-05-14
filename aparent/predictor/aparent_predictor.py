import keras
from keras.models import Sequential, Model, load_model
from keras import backend as K

import tensorflow as tf

import os

import isolearn.keras as iso
import numpy as np

def logit(x) :
	return np.log(x / (1.0 - x))

def get_aparent_encoder() :
	onehot_encoder = iso.OneHotEncoder(205)

	def encode_for_aparent(sequences) :
		one_hots = np.concatenate([np.reshape(onehot_encoder(sequence), (1, len(sequence), 4, 1)) for sequence in sequences], axis=0)

		return [
			one_hots,
			np.zeros((len(sequences), 13)),
			np.ones((len(sequences), 1))
		]

	return encode_for_aparent

def get_aparent_legacy_encoder() :
	onehot_encoder = iso.OneHotEncoder(185)

	def encode_for_aparent(sequences) :
		one_hots = np.concatenate([np.reshape(onehot_encoder(sequence), (1, 1, len(sequence), 4)) for sequence in sequences], axis=0)

		return [
			one_hots,
			np.zeros((len(sequences), 36)),
			np.ones((len(sequences), 1))
		]

	return encode_for_aparent

def get_apadb_encoder() :
	onehot_encoder = iso.OneHotEncoder(205)

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


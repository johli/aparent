import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, LSTM, ConvLSTM2D, GRU, BatchNormalization, LocallyConnected2D, Permute
from keras.layers import Concatenate, Reshape, Softmax, Conv2DTranspose, Embedding, Multiply
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
from keras import backend as K
import keras.losses

import tensorflow as tf

import isolearn.keras as iso

import numpy as np


#APARENT Saved Model definition

def load_saved_predictor(model_path, library_context=None) :

	saved_model = load_model(model_path)

	hardcoded_lib = np.zeros((1, 36))
	if library_context is not None :
		hardcoded_lib[0, library_context] = 1.


	def _initialize_predictor_weights(predictor_model, saved_model=saved_model) :
		#Load pre-trained model
		predictor_model.get_layer('iso_conv_layer_1').set_weights(saved_model.get_layer('iso_conv_layer_1').get_weights())
		predictor_model.get_layer('iso_conv_layer_1').trainable = False

		predictor_model.get_layer('iso_conv_layer_2').set_weights(saved_model.get_layer('iso_conv_layer_2').get_weights())
		predictor_model.get_layer('iso_conv_layer_2').trainable = False

		predictor_model.get_layer('iso_dense_layer_1').set_weights(saved_model.get_layer('iso_dense_layer_1').get_weights())
		predictor_model.get_layer('iso_dense_layer_1').trainable = False

		predictor_model.get_layer('iso_out_layer_1').set_weights(saved_model.get_layer('iso_out_layer_1').get_weights())
		predictor_model.get_layer('iso_out_layer_1').trainable = False

		predictor_model.get_layer('cut_conv_layer_1').set_weights(saved_model.get_layer('cut_conv_layer_1').get_weights())
		predictor_model.get_layer('cut_conv_layer_1').trainable = False

		predictor_model.get_layer('cut_conv_layer_2').set_weights(saved_model.get_layer('cut_conv_layer_2').get_weights())
		predictor_model.get_layer('cut_conv_layer_2').trainable = False

		predictor_model.get_layer('cut_dense_layer_1').set_weights(saved_model.get_layer('cut_dense_layer_1').get_weights())
		predictor_model.get_layer('cut_dense_layer_1').trainable = False

		predictor_model.get_layer('cut_out_layer_1').set_weights(saved_model.get_layer('cut_out_layer_1').get_weights())
		predictor_model.get_layer('cut_out_layer_1').trainable = False

	def _load_predictor_func(sequence_input) :

		#APARENT Legacy parameters
		seq_input_shape = (1, 185, 4)
		#lib_input_shape = (36,)
		#distal_pas_shape = (1,)


		#Isoform model definition
		iso_layer_1 = Conv2D(70, (8, 4), padding='valid', activation='relu', kernel_initializer='zeros', data_format='channels_first', name='iso_conv_layer_1')
		iso_layer_1_pool = MaxPooling2D(pool_size=(2, 1), data_format='channels_first', name='iso_maxpool_layer_1')
		iso_layer_2 = Conv2D(110, (6, 1), padding='valid', activation='relu', kernel_initializer='zeros', data_format='channels_first', name='iso_conv_layer_2')
		iso_layer_dense = Dense(80, activation='relu', kernel_initializer='zeros', name='iso_dense_layer_1')
		iso_layer_drop = Dropout(0.2, name='iso_drop_layer_1')
		iso_layer_out = Dense(2, activation='linear', kernel_initializer='zeros', name='iso_out_layer_1')

		def iso_model(seq_input, distal_pas_input, lib_input) :
		    return Concatenate()([
	            iso_layer_drop(
	                iso_layer_dense(
	                    Concatenate()([
	                        Flatten()(
	                            iso_layer_2(
	                                iso_layer_1_pool(
	                                    iso_layer_1(
	                                        seq_input
	                                    )
	                                )
	                            )
	                        ),
	                        distal_pas_input
	                    ])
	                ), training=False
	            ),
	            lib_input
	        ])

		#Cut model definition
		cut_layer_1 = Conv2D(70, (8, 4), padding='valid', activation='relu', kernel_initializer='zeros', data_format='channels_first', name='cut_conv_layer_1')
		cut_layer_1_pool = MaxPooling2D(pool_size=(2, 1), data_format='channels_first', name='cut_maxpool_layer_1')
		cut_layer_2 = Conv2D(110, (6, 1), padding='valid', activation='relu', kernel_initializer='zeros', data_format='channels_first', name='cut_conv_layer_2')
		cut_layer_dense = Dense(400, activation='relu', kernel_initializer='zeros', name='cut_dense_layer_1') #200 if _pasaligned
		cut_layer_drop = Dropout(0.2, name='cut_drop_layer_1')
		cut_layer_out = Dense(186, activation='linear', kernel_initializer='zeros', name='cut_out_layer_1')

		def cut_model(seq_input, distal_pas_input, lib_input) :
		    return Concatenate()([
	            cut_layer_drop(
	                cut_layer_dense(
	                    Concatenate()([
	                        Flatten()(
	                            cut_layer_2(
	                                cut_layer_1_pool(
	                                    cut_layer_1(
	                                        seq_input
	                                    )
	                                )
	                            )
	                        ),
	                        distal_pas_input
	                    ])
	                ), training=False
	            ),
	            lib_input
	        ])

		sequence_input_flipped = Lambda(lambda x: K.permute_dimensions(x, (0, 3, 1, 2)), output_shape=(1, 185, 4))(sequence_input)

		#lib_input = Input(tensor=K.zeros(lib_input_shape))
		lib_input = Lambda(lambda x: K.tile(K.variable(hardcoded_lib), (K.shape(x)[0], 1)))(sequence_input_flipped)

		#distal_pas_input = Input(tensor=K.ones(distal_pas_shape))
		distal_pas_input = Lambda(lambda x: K.tile(K.variable(np.ones((1, 1))), (K.shape(x)[0], 1)))(sequence_input_flipped)

		out_iso_dense = iso_model(sequence_input_flipped, distal_pas_input, lib_input)
		score_iso = iso_layer_out(out_iso_dense)
		out_iso = Softmax(axis=-1)(score_iso)

		out_cut_dense = cut_model(sequence_input_flipped, distal_pas_input, lib_input)
		score_cut = cut_layer_out(out_cut_dense)
		out_cut = Softmax(axis=-1)(score_cut)

		score_iso_trimmed = Lambda(lambda x: K.expand_dims(x[:, 1], axis=-1), output_shape=(1,))(score_iso)
		out_iso_trimmed = Lambda(lambda x: K.expand_dims(x[:, 1], axis=-1), output_shape=(1,))(out_iso)

		
		predictor_inputs = []#[lib_input, distal_pas_input]
		predictor_outputs = [out_iso_trimmed, out_cut, score_iso_trimmed, score_cut, out_iso_dense]

		return predictor_inputs, predictor_outputs, _initialize_predictor_weights

	return _load_predictor_func

from __future__ import print_function
import keras
from keras import backend as K
import keras.losses

import tensorflow as tf

#Keras loss functions

def get_sample_weights(counts, batch_size) :
    weights = K.log(1.0 + counts) / K.log(2.0)
    
    return (weights / K.sum(weights)) * batch_size

def get_cross_entropy(batch_size, use_sample_weights=False) :
    
    def cross_entropy(inputs) :
        y_true, y_pred, counts = inputs
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        
        sample_weights = tf.ones((batch_size,))
        if use_sample_weights :
            sample_weights = get_sample_weights(counts, batch_size)

        return -K.sum(y_true * K.log(y_pred), axis=-1) * sample_weights
    
    return cross_entropy

def get_mean_cross_entropy(batch_size, use_sample_weights=False) :
    
    def mean_cross_entropy(inputs) :
        y_true, y_pred, counts = inputs
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        
        sample_weights = tf.ones((batch_size,))
        if use_sample_weights :
            sample_weights = get_sample_weights(counts, batch_size)

        return -K.mean(y_true * K.log(y_pred), axis=-1) * sample_weights
    
    return mean_cross_entropy

def get_sigmoid_entropy(batch_size, use_sample_weights=False) :
    
    def sigmoid_entropy(inputs) :
        y_true, y_pred, counts = inputs
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        
        sample_weights = tf.ones((batch_size,))
        if use_sample_weights :
            sample_weights = get_sample_weights(counts, batch_size)

        return -K.sum(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred), axis=-1) * sample_weights
    
    return sigmoid_entropy

def get_mean_sigmoid_entropy(batch_size, use_sample_weights=False) :
    
    def mean_sigmoid_entropy(inputs) :
        y_true, y_pred, counts = inputs
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        
        sample_weights = tf.ones((batch_size,))
        if use_sample_weights :
            sample_weights = get_sample_weights(counts, batch_size)

        return -K.mean(y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred), axis=-1) * sample_weights
    
    return mean_sigmoid_entropy

def get_kl_divergence(batch_size, use_sample_weights=False) :
    
    def kl_divergence(inputs) :
        y_true, y_pred, counts = inputs
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())
        
        sample_weights = tf.ones((batch_size,))
        if use_sample_weights :
            sample_weights = get_sample_weights(counts, batch_size)

        return K.sum(y_true * K.log(y_true / y_pred), axis=-1) * sample_weights
    
    return kl_divergence

def get_margin_kl_divergence(batch_size, use_sample_weights=False) :

    def margin_kl_divergence(inputs) :
        y_true, y_pred, counts = inputs
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())

        margins = 1. / (2. * counts * 0.5)
        y_true_lo = K.maximum(y_true - margins, K.constant(0.0, shape=(1, 1)))
        y_true_hi = K.minimum(y_true + margins, K.constant(1.0, shape=(1, 1)))

        kl_div = K.switch((y_pred > y_true_lo) & (y_pred < y_true_hi), K.zeros_like(y_true), y_true * K.log(y_true / y_pred))

        return K.sum(kl_div, axis=-1)
    
    return margin_kl_divergence

def get_mean_kl_divergence(batch_size, use_sample_weights=False) :
    
    def mean_kl_divergence(inputs) :
        y_true, y_pred, counts = inputs
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())
        
        sample_weights = tf.ones((batch_size,))
        if use_sample_weights :
            sample_weights = get_sample_weights(counts, batch_size)

        return K.mean(y_true * K.log(y_true / y_pred), axis=-1) * sample_weights
    
    return mean_kl_divergence

def get_sigmoid_kl_divergence(batch_size, use_sample_weights=False) :
    
    def sigmoid_kl_divergence(inputs) :
        y_true, y_pred, counts = inputs
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())
        
        sample_weights = tf.ones((batch_size,))
        if use_sample_weights :
            sample_weights = get_sample_weights(counts, batch_size)

        return K.sum(y_true * K.log(y_true / y_pred) + (1.0 - y_true) * K.log((1.0 - y_true) / (1.0 - y_pred)), axis=-1) * sample_weights
    
    return sigmoid_kl_divergence

def get_margin_sigmoid_kl_divergence(batch_size, use_sample_weights=False) :
    
    def margin_sigmoid_kl_divergence(inputs) :
        y_true, y_pred, counts = inputs
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())

        margins = 1. / (2. * counts * 0.5)
        y_true_lo = K.maximum(y_true - margins, K.constant(0.0, shape=(1, 1)))
        y_true_hi = K.minimum(y_true + margins, K.constant(1.0, shape=(1, 1)))

        kl_div = y_true * K.log(y_true / y_pred) + (1.0 - y_true) * K.log((1.0 - y_true) / (1.0 - y_pred))

        margin_loss = K.switch((y_pred > y_true_lo) & (y_pred < y_true_hi), K.zeros_like(kl_div), kl_div)

        return K.sum(margin_loss, axis=-1)
    
    return margin_sigmoid_kl_divergence

def get_mean_sigmoid_kl_divergence(batch_size, use_sample_weights=False) :
    
    def mean_sigmoid_kl_divergence(inputs) :
        y_true, y_pred, counts = inputs
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())
        
        sample_weights = tf.ones((batch_size,))
        if use_sample_weights :
            sample_weights = get_sample_weights(counts, batch_size)

        return K.mean(y_true * K.log(y_true / y_pred) + (1.0 - y_true) * K.log((1.0 - y_true) / (1.0 - y_pred)), axis=-1) * sample_weights
    
    return mean_sigmoid_kl_divergence

def get_symmetric_kl_divergence(batch_size, use_sample_weights=False) :
    
    def symmetric_kl_divergence(inputs) :
        y_true, y_pred, counts = inputs
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())
        
        sample_weights = tf.ones((batch_size,))
        if use_sample_weights :
            sample_weights = get_sample_weights(counts, batch_size)

        return K.sum(y_true * K.log(y_true / y_pred), axis=-1) * sample_weights + K.sum(y_pred * K.log(y_pred / y_true), axis=-1) * sample_weights
    
    return symmetric_kl_divergence

def get_mean_symmetric_kl_divergence(batch_size, use_sample_weights=False) :
    
    def mean_symmetric_kl_divergence(inputs) :
        y_true, y_pred, counts = inputs
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())
        
        sample_weights = tf.ones((batch_size,))
        if use_sample_weights :
            sample_weights = get_sample_weights(counts, batch_size)

        return K.mean(y_true * K.log(y_true / y_pred), axis=-1) * sample_weights + K.mean(y_pred * K.log(y_pred / y_true), axis=-1) * sample_weights
    
    return mean_symmetric_kl_divergence

def get_symmetric_sigmoid_kl_divergence(batch_size, use_sample_weights=False) :
    
    def symmetric_sigmoid_kl_divergence(inputs) :
        y_true, y_pred, counts = inputs
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())
        
        sample_weights = tf.ones((batch_size,))
        if use_sample_weights :
            sample_weights = get_sample_weights(counts, batch_size)

        return K.sum(y_true * K.log(y_true / y_pred) + (1.0 - y_true) * K.log((1.0 - y_true) / (1.0 - y_pred)), axis=-1) * sample_weights + K.sum(y_pred * K.log(y_pred / y_true) + (1.0 - y_pred) * K.log((1.0 - y_pred) / (1.0 - y_true)), axis=-1) * sample_weights
    
    return symmetric_sigmoid_kl_divergence

def get_mean_symmetric_sigmoid_kl_divergence(batch_size, use_sample_weights=False) :
    
    def mean_symmetric_sigmoid_kl_divergence(inputs) :
        y_true, y_pred, counts = inputs
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())
        
        sample_weights = tf.ones((batch_size,))
        if use_sample_weights :
            sample_weights = get_sample_weights(counts, batch_size)

        return K.mean(y_true * K.log(y_true / y_pred) + (1.0 - y_true) * K.log((1.0 - y_true) / (1.0 - y_pred)), axis=-1) * sample_weights + K.mean(y_pred * K.log(y_pred / y_true) + (1.0 - y_pred) * K.log((1.0 - y_pred) / (1.0 - y_true)), axis=-1) * sample_weights
    
    return mean_symmetric_sigmoid_kl_divergence


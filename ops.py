
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import utils 

def downscale2d(x, data_format, factor=2):
    with tf.variable_scope('Downscale2D'):
        assert isinstance(factor, int) and factor >= 1
        if factor == 1: return x
        if data_format == 'NHWC':
            ksize = [1, factor, factor, 1]
        else:
            ksize = [1, 1, factor, factor]
        x = tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format=data_format)
        return x

def upscale2d(x, data_format, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        if data_format == 'NHWC':
            s = x.shape
            x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
            x = tf.tile(x, [1, 1, factor, 1, factor, 1])
            x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
        else:
            s = x.shape
            x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
            x = tf.tile(x, [1, 1, 1, factor, 1, factor])
            x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x

def lerp(a, b, t): 
    return a + (b - a) * t

def leaky_relu(inputs, alpha):
    return tf.nn.leaky_relu(inputs, alpha=alpha)

def lerp_clip(a, b, t): 
    return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)

def relu(inputs):
    return tf.nn.relu(inputs)

def attention(x):
    with tf.variable_scope('Attention'):
        x1, x2 = tf.split(x, 2, 3)    
        x2 = tf.sigmoid(x2)
        x = tf.multiply(x1,x2)        
        return x

def to_rgb(name, x, data_format):
    x = conv2d(name, x, 3, 1, data_format)    
    return x

def from_rgb(name, x, filters, data_format):
    with tf.variable_scope(name):
        x = conv2d('conv', x, filters, 1, data_format)
        x = attention(x)
    return x

def sample(mean, std):
    shape = utils.int_shape(mean)
    with tf.variable_scope('Sample'):
        n = tf.random_normal([shape[0], shape[1]])        
    return mean + tf.multiply(n, std)
    
def squash(x, data_format, epsilon=1e-8):
    with tf.variable_scope('Squash'):
        shape = utils.int_shape(x)
        if len(shape) == 2:
            axis = 1
        else:
            axis = 3 if data_format == 'NHWC' else 1
        squared_norm = tf.reduce_sum(tf.square(x), axis=axis, keepdims=True)
        scalar_factor = squared_norm / (1 + squared_norm) * tf.rsqrt(squared_norm + epsilon)
        return x * scalar_factor

def pixel_norm(x, data_format, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        shape = utils.int_shape(x)
        if len(shape) == 2:
            axis = 1
        else:
            axis = 3 if data_format == 'NHWC' else 1
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=axis, keepdims=True) + epsilon)

def apply_bias(x, data_format):
    shape = utils.int_shape(x)
    assert(len(shape)==2 or len(shape)==4)
    if len(shape) == 2:        
        channels = shape[1]
    else:        
        channels = shape[3] if data_format == 'NHWC' else shape[1]
    b = tf.get_variable('bias', shape=[channels], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        if data_format == 'NHWC':
            return x + tf.reshape(b, [1, 1, 1, -1])
        else:
            return x + tf.reshape(b, [1, -1, 1, 1])            

def dense(name, x, fmaps, data_format, gain=np.sqrt(2), use_wscale=True, has_bias=True, use_tanh=False):
    with tf.variable_scope(name):
        if len(x.shape) > 2:
            x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
        w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale, use_tanh=use_tanh)
        w = tf.cast(w, x.dtype)
        x = tf.matmul(x, w)     
        if has_bias:
            x = apply_bias(x, data_format)
        return x

def get_weight(shape, gain=np.sqrt(2), use_wscale=True, fan_in=None, use_tanh=False):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in) # He init
    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        w = tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal())
        if use_tanh:
            w = tf.tanh(w)
        w = w * wscale
        return w
    else:
        w = tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))
        if use_tanh:
            w = tf.tanh(w)
        return w

def conv2d(name, x, fmaps, kernel, data_format, has_bias=True, gain=np.sqrt(2), use_wscale=True, use_tanh=False):
    with tf.variable_scope(name):
        assert kernel >= 1 and kernel % 2 == 1
        w = get_weight([kernel, kernel, x.shape[3 if data_format == 'NHWC' else 1].value, fmaps], gain=gain, use_wscale=use_wscale, use_tanh=use_tanh)
        w = tf.cast(w, x.dtype)
        x = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format=data_format)
        if has_bias:
            x = apply_bias(x, data_format)
        return x

def conv2d_down(name, x, fmaps, kernel, data_format, gain=np.sqrt(2), has_bias=True, use_wscale=True, use_tanh=False):
    with tf.variable_scope(name):
        assert kernel >= 1 and kernel % 2 == 1
        w = get_weight([kernel, kernel, x.shape[3 if data_format == 'NHWC' else 1].value, fmaps], gain=gain, use_wscale=use_wscale, use_tanh=use_tanh)
        w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
        w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
        w = tf.cast(w, x.dtype)
        x = tf.nn.conv2d(x, w, strides=[1,2,2,1] if data_format=='NHWC' else [1,1,2,2], padding='SAME', data_format=data_format)
        if has_bias:
            x = apply_bias(x, data_format)
        return x

def conv2d_up(name, x, fmaps, kernel, data_format, gain=np.sqrt(2), has_bias=True, use_wscale=True):
    with tf.variable_scope(name):
        assert kernel >= 1 and kernel % 2 == 1
        c = x.shape[3 if data_format == 'NHWC' else 1].value
        w = get_weight([kernel, kernel, fmaps, c], gain=gain, use_wscale=use_wscale, fan_in=(kernel**2)*c)
        w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
        w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
        w = tf.cast(w, x.dtype)
        if data_format == 'NHWC':
            os = [tf.shape(x)[0], x.shape[1] * 2, x.shape[2] * 2, fmaps]
        else:
            os = [tf.shape(x)[0], fmaps, x.shape[2] * 2, x.shape[3] * 2]
        x = tf.nn.conv2d_transpose(x, w, os, strides=[1,2,2,1] if data_format=='NHWC' else [1,1,2,2], padding='SAME', data_format=data_format)
        if has_bias:
            x = apply_bias(x, data_format)
        return x

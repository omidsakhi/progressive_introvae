import tensorflow as tf
import ops
import utils

def fn(resolution):
    res_to_fn = {
        2: 512,
        4: 512,
        8: 256,
        16: 256,
        32: 256,
        64: 128,
        128: 64
    }
    return res_to_fn[resolution]


def rname(resolution):
    return str(resolution) + 'x' + str(resolution)


def block_up(x, filters, kernel_size, name):
  with tf.variable_scope(name):
    x = ops.conv2d_up("conv_up", x, filters, 1, 'NHWC')
    x = ops.attention(x)
    x = ops.conv2d("conv_2", x, filters, kernel_size, 'NHWC')
    x = ops.attention(x)
    x = ops.conv2d("conv_3", x, filters, kernel_size, 'NHWC')
    x = ops.attention(x)
    return x


def block_dn(x, fn, fn_last, kernel_size, name):
  with tf.variable_scope(name):
    x = ops.conv2d("conv_1", x, fn, kernel_size, 'NHWC')
    x = ops.attention(x)
    x = ops.conv2d("conv_2", x, fn, kernel_size, 'NHWC')
    x = ops.attention(x)
    x = ops.conv2d_down("conv_dn", x, fn_last, 1, 'NHWC')
    x = ops.attention(x)
    return x


def enc(x, start_res, end_res, scope='Encoder'):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    res = end_res       
    if res > start_res:
      x1 = ops.downscale2d(x, 'NHWC')
      x1 = ops.from_rgb('rgb_' + rname(res // 2), x1, fn(res//2), 'NHWC')
      x2 = ops.from_rgb('rgb_' + rname(res), x, fn(res // 2), 'NHWC')
      t = tf.get_variable(rname(res)+'_t', shape=[], dtype=tf.float32, collections=[tf.GraphKeys.GLOBAL_VARIABLES,"lerp"],
                          initializer=tf.zeros_initializer(), trainable=False)                
      x2 = block_dn(x2, fn(res), fn(res // 2), 3, rname(res))
      x = ops.lerp_clip(x1, x2, t)
      res = res // 2
    else:
      x = ops.from_rgb('rgb_' + rname(res), x, fn(res), 'NHWC')

    while res >= 4:          
      x = block_dn(x, fn(res), fn(res // 2), 3, rname(res))
      res = res // 2        
      
    x = tf.layers.flatten(x)
    x = ops.dense('fc1', x, 512, 'NHWC')
    mean, std = tf.split(x, 2, 1)            
    return mean, std


def dec(x, start_res, end_res, scope='Decoder'):
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    x = ops.dense('fc1', x, fn(4) * 4 * 4, 'NHWC')
    x = tf.reshape(x, [-1, 4, 4, fn(4)])    
    res = 8
    prev_x = None
    while res <= end_res:
        prev_x = x
        x = block_up(x, fn(res), 3, rname(res))
        res *= 2
    res = res // 2

    if res > start_res:
        t = tf.get_variable(rname(res) + '_t', shape=[], collections=[tf.GraphKeys.GLOBAL_VARIABLES,"lerp"],
                            dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=False)
        x1 = ops.to_rgb('rgb_'+rname(res // 2), prev_x, 'NHWC')
        x1 = ops.upscale2d(x1, 'NHWC')
        x2 = ops.to_rgb('rgb_'+rname(res), x, 'NHWC')
        x = ops.lerp_clip(x1, x2, t)
    else:
        x = ops.to_rgb('rgb_'+rname(res), x, "NHWC")
    
    x_shape = utils.int_shape(x)
    assert(end_res == x_shape[1])
    assert(end_res == x_shape[2])        
    return x

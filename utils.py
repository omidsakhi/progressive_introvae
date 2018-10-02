import tensorflow as tf
from PIL import Image
import math
import numpy as np


def int_shape(x):
    if str(x.get_shape()[0]) != '?':
        return list(map(int, x.get_shape()))
    return [-1]+list(map(int, x.get_shape()[1:]))

def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

def write_images(images, filename, data_format):
    sq = math.floor(math.sqrt(len(images)))
    assert sq ** 2 == len(images)
    sq = int(sq)
    if data_format == 'NCHW':
        images = [np.transpose(img,[1,2,0]) for img in images]
    image_rows = [np.concatenate(images[i:i+sq], axis=0)
                  for i in range(0, len(images), sq)]
    tiled_image = np.concatenate(image_rows, axis=1)
    tiled_image = (tiled_image + 1.0) / 2.0 * 255.0
    tiled_image = np.clip(tiled_image, 0, 255)
    img = Image.fromarray(np.uint8(tiled_image), mode='RGB')
    file_obj = tf.gfile.Open(filename, 'w')
    img.save(file_obj, format='png')


def optimistic_restore(session, ckpt, graph):
    reader = tf.train.NewCheckpointReader(ckpt)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    for var_name, saved_var_name in var_names:
        curr_var = graph.get_tensor_by_name(var_name)
        var_shape = curr_var.get_shape().as_list()
        if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(curr_var)
    opt_saver = tf.train.Saver(restore_vars, name='opt_saver')
    opt_saver.restore(session, ckpt)


def restore(sess, restore_dir):
    if restore_dir:
        latest_checkpoint = tf.train.latest_checkpoint(restore_dir)
        if latest_checkpoint:
            optimistic_restore(sess, latest_checkpoint, tf.get_default_graph())
            return latest_checkpoint
    return None

def assert_resolution_step(resolution_step_tensor):
  if not resolution_step_tensor.dtype.base_dtype.is_integer:
    raise TypeError('Existing "resolution_step" does not have integer type: %s' %
                    resolution_step_tensor.dtype)

  if (resolution_step_tensor.get_shape().ndims != 0 and
      resolution_step_tensor.get_shape().is_fully_defined()):
    raise TypeError('Existing "resolution_step" is not scalar: %s' %
                    resolution_step_tensor.get_shape())

def get_resolution_step(graph=None):
    graph = graph or tf.get_default_graph()    
    resolution_step_tensor = None
    resolution_step_tensors = graph.get_collection("resolution_step")
    if len(resolution_step_tensors) == 1:
        resolution_step_tensor = resolution_step_tensors[0]
    elif not resolution_step_tensors:
        try:
            resolution_step_tensor = graph.get_tensor_by_name('resolution_step:0')
        except KeyError:
            return None
    else:
        raise TypeError('Multiple tensors in resolution_step collection.')        
    assert_resolution_step(resolution_step_tensor)
    return resolution_step_tensor

def create_resolution_step(graph=None):
  graph = graph or tf.get_default_graph()
  if get_resolution_step(graph) is not None:
    raise ValueError('"resolution_step" already exists.')
  with graph.as_default() as g, g.name_scope(None):
    return tf.get_variable(
        "resolution_step",
        shape=[],
        dtype=tf.int64,
        initializer=tf.zeros_initializer(),
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                     "resolution_step"])

def get_or_create_resolution_step(graph=None):
    graph = graph or tf.get_default_graph()
    resolution_step_tensor = get_resolution_step(graph)
    if resolution_step_tensor is None:
        resolution_step_tensor = create_resolution_step(graph)
    return resolution_step_tensor

def reset_resolution_step(sess=None):
    sess = sess or tf.get_default_session()
    resolution_step_tensor = get_or_create_resolution_step(sess.graph)
    op = tf.assign(resolution_step_tensor, 0)
    sess.run(op)

def print_layers(scope, hide_layers_with_no_params=False):    
    
    print ()
    print (scope, ' ---> ')
    print ()    
    total_params = 0
    for v in tf.trainable_variables():
        name = v.name
        if scope in v.name:            
            name = name.replace(scope+'/', '')
            name = name.replace(':0', '')
            print ('%-32s' % name, v.shape)
            prod = 1
            for dim in v.shape:                
                prod *= dim
            total_params += prod
    print ()
    print ('<--- Total Parameters: ', total_params)
    print ()

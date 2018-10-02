from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, ops, utils

# Standard Imports
from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import
import numpy as np
import tensorflow as tf
from PIL import Image
import input_pipelines
import models
from tensorflow.contrib.tpu.python.tpu import tpu_config  # pylint: disable=E0611
from tensorflow.contrib.tpu.python.tpu import tpu_estimator  # pylint: disable=E0611
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer  # pylint: disable=E0611
from tensorflow.python.estimator import estimator  # pylint: disable=E0611

FLAGS = flags.FLAGS

global dataset
dataset = input_pipelines

USE_TPU = False
DRY_RUN = False

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'tpu', default='omid-sakhi',
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')
flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

flags.DEFINE_string('data_dir', 'gs://os_celeba/dataset' if USE_TPU else 'C:/Projects/datasets/tfr-celeba128',
                    'Bucket/Folder that contains the data tfrecord files')
flags.DEFINE_string(
    'model_dir', 'gs://os_celeba/output1' if USE_TPU else './output', 'Output model directory')
flags.DEFINE_integer('noise_dim', 256,
                     'Number of dimensions for the noise vector')
flags.DEFINE_integer('batch_size', 128 if USE_TPU else 32,
                     'Batch size for both generator and discriminator')
flags.DEFINE_integer('start_resolution', 8,
                     'Starting resoltuion')
flags.DEFINE_integer('end_resolution', 128,
                     'Ending resoltuion')
flags.DEFINE_integer('resolution_steps', 10000 if not DRY_RUN else 60,
                     'Resoltuion steps')
flags.DEFINE_integer('num_shards', 8, 'Number of TPU chips')
flags.DEFINE_integer('train_steps', 500000, 'Number of training steps')
flags.DEFINE_integer('train_steps_per_eval', 5000 if USE_TPU else (200 if not DRY_RUN else 20) ,
                     'Steps per eval and image generation')
flags.DEFINE_integer('iterations_per_loop', 500 if USE_TPU else (50 if not DRY_RUN else 5) ,
                     'Steps per interior TPU loop. Should be less than'
                     ' --train_steps_per_eval')
flags.DEFINE_float('learning_rate', 0.001, 'LR for both D and G')
flags.DEFINE_boolean('eval_loss', False,
                     'Evaluate discriminator and generator loss during eval')
flags.DEFINE_boolean('use_tpu', True if USE_TPU else False,
                     'Use TPU for training')
flags.DEFINE_integer('num_eval_images', 100,
                     'Number of images for evaluation')


def lerp_update_ops(resolution, value):
    name = str(resolution) + 'x' + str(resolution)
    gt = tf.get_default_graph().get_tensor_by_name('Decoder/'+name+'_t:0')
    assert(gt is not None)
    dt = tf.get_default_graph().get_tensor_by_name('Encoder/'+name+'_t:0')
    assert(dt is not None)
    return [tf.assign(gt, value), tf.assign(dt, value)]

def model_fn(features, labels, mode, params):
    del labels
    resolution = params['resolution']

    if mode == tf.estimator.ModeKeys.PREDICT:
        ###########
        # PREDICT #
        ###########
        random_noise = features['random_noise']
        predictions = {
            'generated_images': models.dec(random_noise, FLAGS.start_resolution, resolution)
        }

        if FLAGS.use_tpu:
            return tpu_estimator.TPUEstimatorSpec(mode=mode, predictions=predictions)
        else:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    def fLreg(mean, std):  
        return tf.reduce_mean(tf.reduce_sum(1.0 + tf.log(tf.square(std) + 1e-8) - tf.square(mean) - tf.square(std), axis=1)) * (10.0 ** (-np.log2(resolution)))

    def fLae(x1,x2):
        return tf.reduce_mean(tf.squared_difference(x1,x2))
    
    def ng(x):
        return tf.stop_gradient(x)

    resolution_step = utils.get_or_create_resolution_step()    
    fadein_rate = tf.minimum(tf.cast(resolution_step, tf.float32) / float(FLAGS.resolution_steps), 1.0)
    batch_size = params['batch_size']   # pylint: disable=unused-variable
    X = features['real_images']  
    Zmean, Zstd = models.enc(X, FLAGS.start_resolution, resolution)    
    Z = ops.sample(Zmean, Zstd)
    Zp = features['random_noise_1']
    Xr = models.dec(Z, FLAGS.start_resolution, resolution)
    Xp = models.dec(Zp, FLAGS.start_resolution, resolution)
    Lae = tf.reduce_mean(fLae(Xr,X))
    Zr = models.enc(ng(Xr), FLAGS.start_resolution, resolution)
    Zpp = models.enc(ng(Xp), FLAGS.start_resolution, resolution)    
    m = 90
    enc_zr = tf.nn.relu(m - fLreg(Zr[0],Zr[1]))
    enc_zpp = tf.nn.relu(m - fLreg(Zpp[0], Zpp[1]))    
    enc_loss = fLreg(Zmean, Zstd) + (enc_zr + enc_zpp) + Lae
    Zr = models.enc(Xr, FLAGS.start_resolution, resolution)
    Zpp = models.enc(Xp, FLAGS.start_resolution, resolution)
    rec_zr = fLreg(Zr[0],Zr[1])
    rec_zpp = fLreg(Zpp[0], Zpp[1])    
    dec_loss = (rec_zr + rec_zpp) + Lae
    
    with tf.variable_scope('Penalties'):
        tf.summary.scalar('enc_loss', tf.reduce_mean(enc_loss))
        tf.summary.scalar('dec_loss', tf.reduce_mean(dec_loss))        
        tf.summary.scalar('mean', tf.reduce_mean(Zmean))
        tf.summary.scalar('std', tf.reduce_mean(Zstd))
        tf.summary.scalar('lae', tf.reduce_mean(Lae))
        tf.summary.scalar('rec_zr', tf.reduce_mean(rec_zr))
        tf.summary.scalar('rec_zpp', tf.reduce_mean(rec_zpp))
        tf.summary.scalar('enc_zr', tf.reduce_mean(enc_zr))
        tf.summary.scalar('enc_zpp', tf.reduce_mean(enc_zpp))
    

    if mode == tf.estimator.ModeKeys.TRAIN or mode == 'RESOLUTION_CHANGE':
        #########
        # TRAIN #
        #########
        e_optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate, beta1=0.9, beta2=0.999)
        d_optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate, beta1=0.9, beta2=0.999)

        if FLAGS.use_tpu:
            e_optimizer = tpu_optimizer.CrossShardOptimizer(e_optimizer)
            d_optimizer = tpu_optimizer.CrossShardOptimizer(d_optimizer)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            e_step = e_optimizer.minimize(
                enc_loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope='Encoder'))
            d_step = d_optimizer.minimize(
                dec_loss,
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope='Decoder'))
            with tf.control_dependencies([e_step, d_step]):       
                increment_global_step = tf.assign_add(
                    tf.train.get_or_create_global_step(), 1)
                increment_resolution_step = tf.assign_add(
                    utils.get_or_create_resolution_step(), 1)
            if resolution>=FLAGS.start_resolution * 2:
                with tf.control_dependencies([increment_global_step, increment_resolution_step]):
                    lerp_ops = lerp_update_ops(resolution, fadein_rate)          
                    joint_op = tf.group([d_step, e_step, lerp_ops[0], lerp_ops[1], increment_global_step, increment_resolution_step])
            else:
                joint_op = tf.group([d_step, e_step, increment_global_step, increment_resolution_step])

            if mode == 'RESOLUTION_CHANGE':
                return [d_optimizer, e_optimizer]
            else:
                if FLAGS.use_tpu:
                    return tpu_estimator.TPUEstimatorSpec(
                        mode=mode,
                        loss=dec_loss + enc_loss,
                        train_op=joint_op)
                else:
                    return tf.estimator.EstimatorSpec(
                        mode=mode,
                        loss=dec_loss + enc_loss,
                        train_op=joint_op)    
    elif mode == tf.estimator.ModeKeys.EVAL:
        ########
        # EVAL #
        ########
        if FLAGS.use_tpu:
            def _eval_metric_fn(e_loss, d_loss):
                # When using TPUs, this function is run on a different machine than the
                # rest of the model_fn and should not capture any Tensors defined there
                return {
                    'enc_loss': tf.metrics.mean(e_loss),
                    'dec_loss': tf.metrics.mean(d_loss)}
            return tpu_estimator.TPUEstimatorSpec(
                mode=mode,
                loss=tf.reduce_mean(enc_loss + enc_loss),
                eval_metrics=(_eval_metric_fn, [enc_loss, dec_loss]))
        else:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=tf.reduce_mean(enc_loss + dec_loss),
                eval_metric_ops={
                    'enc_loss': tf.metrics.mean(enc_loss),
                    'dec_loss': tf.metrics.mean(dec_loss)
                })

    raise ValueError('Invalid mode provided to model_fn')


def noise_input_fn(params):
    np.random.seed(0)
    noise_dataset = tf.data.Dataset.from_tensors(tf.constant(
        np.random.randn(params['batch_size'], FLAGS.noise_dim), dtype=tf.float32))
    noise = noise_dataset.make_one_shot_iterator().get_next()
    return {'random_noise': noise}, None

def get_estimator(model_dir, resolution):
    tpu_cluster_resolver = None

    if FLAGS.use_tpu:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu,
            zone=FLAGS.tpu_zone,
            project=FLAGS.gcp_project)

        config = tpu_config.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=model_dir,
            tpu_config=tpu_config.TPUConfig(
                num_shards=FLAGS.num_shards,
                iterations_per_loop=FLAGS.iterations_per_loop))

        est = tpu_estimator.TPUEstimator(
            model_fn=model_fn,
            use_tpu=FLAGS.use_tpu,
            config=config,
            params={"data_dir": FLAGS.data_dir, "resolution": resolution},
            train_batch_size=FLAGS.batch_size,
            eval_batch_size=FLAGS.batch_size)

        local_est = tpu_estimator.TPUEstimator(
            model_fn=model_fn,
            use_tpu=False,
            config=config,
            params={"data_dir": FLAGS.data_dir, "resolution": resolution},
            predict_batch_size=FLAGS.num_eval_images)
    else:
        est = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=model_dir,
            params={"data_dir": FLAGS.data_dir, "batch_size": FLAGS.batch_size, "resolution": resolution})

        local_est = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=model_dir,
            params={"data_dir": FLAGS.data_dir, "batch_size": FLAGS.num_eval_images, "resolution": resolution})
    return est, local_est

def change_resolution(resolution):
    batch_size = 1
    graph = tf.Graph()
    store_dir = os.path.join(FLAGS.model_dir, 'resolution_' + str(resolution))
    restore_dir = os.path.join(FLAGS.model_dir, 'resolution_' + str(resolution // 2))
    tf.gfile.MakeDirs(store_dir)
    ckpt_file = store_dir + '/model.ckp'    
    with graph.as_default(): # pylint: disable=E1129
        train_input = dataset.TrainInputFunction(FLAGS.noise_dim, resolution, 'NHWC')    
        params = {'data_dir' : FLAGS.data_dir, 'batch_size' : batch_size , "resolution": resolution}
        features, labels = train_input(params)
        optimizers = model_fn(features, labels, 'RESOLUTION_CHANGE', params)        
        global_step = tf.train.get_or_create_global_step()        
        with tf.Session() as sess:                          
            sess.run(tf.global_variables_initializer())
            utils.restore(sess, restore_dir)
            utils.reset_resolution_step()
            for opt in optimizers:
                sess.run(tf.variables_initializer(opt.variables()))
            saver = tf.train.Saver(name='main_saver')                                     
            saver.save(sess, ckpt_file, global_step = global_step)

def main(argv):

    del argv

    tf.gfile.MakeDirs(os.path.join(FLAGS.model_dir))    

    resolution = FLAGS.end_resolution 
    initial_checkpoint = None
    while initial_checkpoint is None and resolution != 1:
        model_dir = os.path.join(FLAGS.model_dir, 'resolution_' + str(resolution))
        initial_checkpoint = tf.train.latest_checkpoint(model_dir)
        resolution = resolution // 2
    if initial_checkpoint is None or resolution == 1:
        resolution = FLAGS.start_resolution    
        model_dir = os.path.join(FLAGS.model_dir, 'resolution_' + str(resolution))
    else:
        resolution *= 2
        model_dir = os.path.join(FLAGS.model_dir, 'resolution_' + str(resolution))
    
    est, local_est = get_estimator(model_dir, resolution)

    current_step = estimator._load_global_step_from_checkpoint_dir(
        model_dir)   # pylint: disable=protected-access,line-too-long

    tf.logging.info('Starting training for %d steps, current step: %d' %
                    (FLAGS.train_steps, current_step))
    while current_step < FLAGS.train_steps:
        if current_step != 0 and current_step % FLAGS.resolution_steps == 0 and resolution != FLAGS.end_resolution:
            resolution *= 2
            tf.logging.info('Change of resolution from %d to %d' % (resolution // 2, resolution))
            model_dir = os.path.join(FLAGS.model_dir, 'resolution_' + str(resolution))
            change_resolution(resolution)
            est, local_est = get_estimator(model_dir, resolution)
        next_checkpoint = min(current_step + FLAGS.train_steps_per_eval,
                              FLAGS.train_steps)
        est.train(input_fn=dataset.TrainInputFunction(FLAGS.noise_dim, resolution, 'NHWC'),
                  max_steps=next_checkpoint)
        current_step = next_checkpoint
        tf.logging.info('Finished training step %d' % current_step)

        if FLAGS.eval_loss:
            metrics = est.evaluate(input_fn=dataset.TrainInputFunction(FLAGS.noise_dim, resolution, 'NHWC'),
                                   steps=FLAGS.num_eval_images // FLAGS.batch_size)
            tf.logging.info('Finished evaluating')
            tf.logging.info(metrics)

        generated_iter = local_est.predict(input_fn=noise_input_fn)
        images = [p['generated_images'][:, :, :] for p in generated_iter]
        filename = os.path.join(FLAGS.model_dir, '%s-%s.png' % (
            str(current_step).zfill(5), 'x' + str(resolution)))
        utils.write_images(images, filename, 'NHWC')
        tf.logging.info('Finished generating images')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

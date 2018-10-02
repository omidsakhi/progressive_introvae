from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from PIL import Image


def parser(serialized_example, resolution):
    image_key = 'image'
    if resolution != 128:
        image_key += str(resolution)
    features = tf.parse_single_example(
        serialized_example,
        features={
            image_key: tf.FixedLenFeature([], tf.string),
            'labels': tf.FixedLenFeature([], tf.string),
        })
    image = tf.image.decode_jpeg(features[image_key])
    image = tf.reshape(image, [3, resolution * resolution])
    image = tf.cast(image, tf.float32)
    image += tf.random_uniform(image.get_shape(), minval=-0.5, maxval=0.5)
    image = tf.clip_by_value(image, 0, 255)
    image = (image * (2.0 / 255.0)) - 1.0
    # tf.cast(features['labels'], tf.int32)
    labels = tf.constant(-1.0, shape=[40])
    return image, labels

class PredictInputFunction(object):

    def __init__(self, noise_dim, resolution):
        self.resolution = resolution
        self.noise_dim = noise_dim

    def __call__(self, params):
        np.random.seed(0)
        noise_dataset = tf.data.Dataset.from_tensors(tf.constant(
            np.random.randn(params['batch_size'], self.noise_dim), dtype=tf.float32))
        noise = noise_dataset.make_one_shot_iterator().get_next()
        return {'random_noise': noise, 'resolution' : self.resolution}, None

class TrainInputFunction(object):

    def __init__(self, noise_dim, resolution, data_format):        
        self.noise_dim = noise_dim
        self.resolution = resolution        
        self.data_format = data_format

    def __call__(self, params):
        batch_size = params['batch_size']
        data_dir = params['data_dir']
        file_pattern = os.path.join(data_dir, 'data_*')
        dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True)
        dataset = dataset.shuffle(buffer_size=200)
        dataset = dataset.repeat()

        def fetch_dataset(filename):
            dataset = tf.data.TFRecordDataset(
                filename, buffer_size=8 * 1024 * 1024)
            return dataset
        dataset = dataset.apply(tf.contrib.data.parallel_interleave(
            fetch_dataset, cycle_length=8, sloppy=True))
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.prefetch(8)
        dataset = dataset.map(lambda x: parser(x, self.resolution), num_parallel_calls=8)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        images, labels = dataset.make_one_shot_iterator().get_next()
        images = tf.reshape(images, [batch_size, self.resolution, self.resolution, 3])
        images = tf.image.random_flip_left_right(images)
        if self.data_format == 'NCHW':
            images = tf.transpose(images, [0, 3, 1, 2])        
        random_noise_1 = tf.random_normal([batch_size, self.noise_dim])
        random_noise_2 = tf.random_normal([batch_size, self.noise_dim])
        features = {
            'real_images': images,
            'random_noise_1': random_noise_1,
            'random_noise_2': random_noise_2,
            'resolution' : self.resolution
        }

        return features, labels

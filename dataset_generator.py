import csv
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import cv2

image_folder = 'D:/Projects/deepflow/data/celeba128'

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_image(addr):
    img = cv2.imread(addr)
    img = img.astype(np.uint8)
    return img

def raw_image(addr):
	f = open(addr, "rb")
	b = f.read()
	f.close()
	return b

writer = None
file_num = 0

def remove_blanks(a_list):
    new_list = []
    for item in a_list:
        if item != "":
            new_list.append(item)
    return new_list

with open(image_folder + '/list_attr_celeba.txt', newline='') as csvfile:
  csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
  attr_names = []
  for idx1, row in tqdm(enumerate(csvreader)):  	
    if (idx1 == 0):
      writer = tf.python_io.TFRecordWriter('data_' + str(file_num) + '.tfrecords')
    elif (idx1 == 1):
      attr_names = row        
    else:
      if (idx1 % 40000 == 0):
        writer.close()
        file_num = file_num + 1
        writer = tf.python_io.TFRecordWriter('data_' + str(file_num) + '.tfrecords')

      row = remove_blanks(row)        
      img = raw_image(image_folder + '/' + row[0])
      attrs = np.array(list(map(int, row[1:])), np.uint8)
      feature = {
      'name' : _bytes_feature(tf.compat.as_bytes(row[0])),
      'image': _bytes_feature(tf.compat.as_bytes(img)),
      'labels': _bytes_feature(tf.compat.as_bytes(attrs.tostring()))
      }     
      example = tf.train.Example(features=tf.train.Features(feature=feature))    
      writer.write(example.SerializeToString())

writer.close()
sys.stdout.flush()
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import numpy as np
import IPython.display as display
import csv


image = 'test/images/train/IMG_3736.JPG'
image2 = 'test/images/train/IMG_3854.JPG'
display.display(display.Image(image2))
display.display(display.Image(image))


image_string = open(image2, 'rb').read()
print(image_string)


train_labels = 'test/train_labels.csv'
with open(train_labels) as f:
    reader = csv.reader(f)
    rows = [row for row in reader]
    print(reader)
    print(rows)


# 下記の関数を使うと値を tf.Example と互換性の有る型に変換できる

def _bytes_feature(value):
  """string / byte 型から byte_list を返す"""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _str_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """float / double 型から float_list を返す"""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """bool / enum / int / uint 型から Int64_list を返す"""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(image_string, train_labels):
    with open(train_labels) as f:
        reader = csv.reader(f)
        data = [row for row in reader]

    image_shape = tf.image.decode_jpeg(image_string).shape

    feature = {
        'filename': _str_feature(data[1][0].encode()),
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'box/1en/xmin': _int64_feature(int(data[1][4])),
        'image_raw': _bytes_feature(image_string)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

for line in str(image_example(image_string, train_labels)).split('\n'):
    print(line)
print('...')


# write TFRecord
record_file = 'images.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
    tf_example = image_example(image_string, train_labels)
    writer.write(tf_example.SerializeToString())


raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')
print(raw_image_dataset)
image_feature_description = {
    'filename': tf.io.FixedLenFeature([], tf.string),
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    # inference: layer too deep to access
    #'bbox/1en/xmin': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string)
}
def _parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
parsed_image_dataset


for image_features in parsed_image_dataset:
  image_raw = image_features['image_raw'].numpy()
  display.display(display.Image(data=image_raw))






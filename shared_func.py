import sys
import tensorflow as tf
from global_vars import n_mels, t, BATCH_SIZE
import numpy as np

def print_and_exit(msg, exit_code):
	print(msg)
	sys.exit(exit_code)

def decode_record(record):
  example = tf.io.parse_single_example(record, {
    'label': tf.io.FixedLenFeature([], tf.dtypes.int64),
    'spectogram': tf.io.FixedLenFeature([], tf.dtypes.string),
  })

  return example

def calc_dataset_size(dataset):
  count = 0
  for _ in dataset:
    count = count + 1

  return count

def get_dataset(path):
  dataset = tf.data.TFRecordDataset(path)
  size = calc_dataset_size(dataset)
  dataset = dataset.shuffle(size).map(decode_record)

  return dataset.batch(BATCH_SIZE)

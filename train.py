import tensorflow as tf
from tensorflow import keras
import sys, getopt
import os
from global_vars import genres, n_mels, t

help_msg = 'train.py -d <data_dir> -s <model_save_dir>'

def print_and_exit(msg, exit_code):
	print(msg)
	sys.exit(exit_code)

def _decode_record(record):
  example = tf.io.parse_single_example(record, {
    'label': tf.io.FixedLenFeature([], tf.dtypes.int64),
    'spectogram': tf.io.FixedLenFeature([], tf.dtypes.string)
  })
  example['spectogram'] = tf.reshape(example['spectogram'], [n_mels, t])

  return example

def get_train_valid_datasets(data_dir):
  train_tfrecord_path = data_dir + "/train.tfrecord"
  valid_tfrecord_path = data_dir + "/valid.tfrecord"

  train_dataset = tf.data.TFRecordDataset(train_tfrecord_path).map(_decode_record)
  valid_dataset = tf.data.TFRecordDataset(valid_tfrecord_path).map(_decode_record)

  return train_dataset, valid_dataset

def train(data_dir, model_save_dir):
  train_dataset, valid_dataset = get_train_valid_datasets(data_dir)

def main(argv):
  data_dir = "/dataset/data"
  model_save_dir = "/saved_model"

  try:
    opts,_ = getopt.getopt(argv, "h:d:s:",["data_dir=, model_save_dir= "])
  except:
    print_and_exit(help_msg, 2)

  for opt, arg in opts:
    if opt == '-h':
      print_and_exit(help_msg, 0)
    if opt == '-d':
      data_dir = arg
    if opt == '-s':
      model_save_dir = arg
  
  if (not os.path.isdir(data_dir)) or (not os.path.isdir(model_save_dir)):
    print_and_exit("Error while trying to open dirs...", 1)

    train(data_dir, model_save_dir)
    

if __name__ == "__main__":
	main(sys.argv[1:])
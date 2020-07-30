import tensorflow as tf
from tensorflow import keras
import sys, getopt
import os
import math

from model import Inception_ResNet

from global_vars import BATCH_SIZE
from shared_func import print_and_exit, get_dataset, calc_dataset_size

help_msg = "model_evaluation.py -d <data_dir> -p <path>, where path points to a saved model and dir points to dir contaiing eval tfrecord"

def get_eval_dataset(data_dir):
  eval_tfrecord_path = data_dir + "/eval.tfrecord"

  return get_dataset(eval_tfrecord_path)

def load_and_eval_model(path_to_model, data_dir):
  eval_dataset = get_eval_dataset(data_dir)

  net = Inception_ResNet.Inception_ResNet
  net.load_weights(filepath = path_to_model)

  loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  eval_loss = keras.metrics.Mean(name = 'eval_loss')
  eval_acc = keras.metrics.SparseCategoricalAccuracy(name = 'eval_acc')

  eval_batch_cnt = math.ceil(calc_dataset_size(eval_dataset) / BATCH_SIZE)
  curr_batch_count = 0

  for batch in eval_dataset:
    curr_batch_count = curr_batch_count + 1

    spectograms = batch['spectogram'].numpy()
    genres = batch['label'].numpy()

    predictions = net(spectograms, training = False)
    loss = loss_object(y_true = genres, y_pred = predictions)
    
    eval_loss.update_state(values = loss)
    eval_acc.update_state(y_true = genres, y_pred = predictions)
    
    print("------------------------")
    print("Batch:{}/{}, loss:{:4f}, acc:{:4f}".format(curr_batch_count, eval_batch_cnt, 
                                                      eval_loss.result().numpy(), eval_acc.result().numpy()))
    print("------------------------")
  
  print("#############################")
  print("Overall test accuracy:{:2f}".format(eval_acc.result().numpy() * 100))
  print("#############################")


def main(argv):
  path_to_model = ""
  data_dir = "/dataset/data"

  try:
    opts, _ = getopt.getopt(argv, "h:d:p:",["data_dir=, path="])
  except:
    print_and_exit(help_msg, 2)
  
  for opt, arg in opts:
    if opt == '-h':
      print_and_exit(help_msg, 0)
    if opt == '-p':
      path_to_model = arg
  

  if (path_to_model == '' or not os.path.isfile(path_to_model) or not os.path.isdir(data_dir)):
    print_and_exit("Check path... ", 1)

  load_and_eval_model(path_to_model, data_dir)  

if __name__ == '__main__':
  main(sys.argv[1:])
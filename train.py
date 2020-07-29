import tensorflow as tf
from tensorflow import keras
import sys, getopt
import os
import math
import numpy as np
import datetime

from global_vars import genres, n_mels, t, BATCH_SIZE, EPOCHS, LEARNING_RATE, EPSILON
from shared_func import print_and_exit, calc_dataset_size, get_dataset

from model import Inception_ResNet

help_msg = 'train.py -d <data_dir> -s <model_save_dir>'

def get_train_valid_datasets(data_dir):
  train_tfrecord_path = data_dir + "/train.tfrecord"
  valid_tfrecord_path = data_dir + "/valid.tfrecord"

  return get_dataset(train_tfrecord_path), get_dataset(valid_tfrecord_path)

def view_summary(model):
  model.build(input_shape = (None, n_mels, t, 1))
  model.summary()

@tf.function
def train_step(spectogram_batch, genre_batch, model, loss_func, optimizer, train_loss, train_acc):
  with tf.GradientTape() as tape:
    predictions = model(spectogram_batch, training = True)
    if(genre_batch.shape[0] != predictions.shape[0]):
      return
    curr_loss = loss_func(y_true = genre_batch, y_pred = predictions)
  
  gradients = tape.gradient(curr_loss ,model.trainable_variables)
  optimizer.apply_gradients(grads_and_vars = zip(gradients, model.trainable_variables))

  train_loss.update_state(values = curr_loss)
  train_acc.update_state(y_true = genre_batch, y_pred = predictions)

@tf.function
def valid_step(spectogram_batch, genre_batch, model, loss_func, valid_loss, valid_acc):
  predictions = model(spectogram_batch, training = False)
  val_loss = loss_func(y_true = genre_batch, y_pred = predictions)

  valid_loss.update_state(values = val_loss)
  valid_acc.update_state(y_true = genre_batch , y_pred = predictions)

def train(data_dir, model_save_dir):
  train_dataset, valid_dataset = get_train_valid_datasets(data_dir)

  net = Inception_ResNet.Inception_ResNet()
  view_summary(net)

  loss_object = keras.losses.SparseCategoricalCrossentropy()
  optimizer = keras.optimizers.RMSprop(learning_rate = LEARNING_RATE, epsilon = EPSILON, decay = 0.94 / (2 * EPOCHS))

  train_loss = keras.metrics.Mean(name = 'training_loss')
  valid_loss = keras.metrics.Mean(name = 'validation_loss')

  train_acc = keras.metrics.SparseCategoricalAccuracy(name = 'training_acc')
  valid_acc = keras.metrics.SparseCategoricalAccuracy(name = 'validation_acc')

  train_batch_cnt = calc_dataset_size(train_dataset)
  best_valid_acc = -1.0

  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
  valid_log_dir = 'logs/gradient_tape/' + current_time + '/valid'
  train_summary_writer = tf.summary.create_file_writer(train_log_dir)
  valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

  
  for epoch in range(EPOCHS):
    curr_batch_count = 0
    for batch in train_dataset:
      curr_batch_count = curr_batch_count + 1
      
      spectograms = batch['spectogram'].numpy()
      spectograms_reshaped = np.ndarray(shape = (BATCH_SIZE, n_mels, t, 1))

      for i in range(spectograms.shape[0]):
        spectograms_reshaped[i] = np.frombuffer(spectograms[i], dtype=np.float32).reshape([n_mels, t, 1])

      #print(np.frombuffer(spectograms[0], dtype=np.float32).shape)
      genres = batch['label'].numpy()

      train_step(spectograms_reshaped, genres, net, loss_object, optimizer, train_loss, train_acc)
      print("------------------------")
      print("Epoch: {}/{}, batch:{}/{}, loss:{:.4f}, accuracy:{:4f}".format(epoch, EPOCHS,
                                                                      curr_batch_count, train_batch_cnt, 
                                                                      train_loss.result().numpy(), train_acc.result().numpy()))
      print("------------------------")
      with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_acc.result(), step=epoch)


    for valid_batch in valid_dataset:
      valid_spectograms = valid_batch['spectogram'].numpy()
	  spectograms_reshaped = np.ndarray(shape = (BATCH_SIZE, n_mels, t, 1))
	  for i in range(valid_spectograms.shape[0]):
        spectograms_reshaped[i] = np.frombuffer(spectograms[i], dtype=np.float32).reshape([n_mels, t, 1])
      valid_genres = valid_batch['label'].numpy()

      valid_step(spectograms_reshaped, valid_genres, net, loss_object, valid_loss, valid_acc)

    with valid_summary_writer.as_default():
      tf.summary.scalar('loss', valid_loss.result(), step=epoch)
      tf.summary.scalar('accuracy', valid_acc.result(), step=epoch)
 
    print("##########################")
    print("Epoch: {}/{}, train_loss:{:4f}, valid_loss:{:4f}, \n train_acc:{:.4f}, valid_acc:{:4f}".format(epoch, EPOCHS,
                                                                      train_loss.result().numpy(), valid_loss.result().numpy(), 
                                                                      train_acc.result().numpy(), valid_acc.result().numpy()))
    print("##########################")

    if(best_valid_acc == -1.0 or np.isclose(train_acc.result().numpy(),best_valid_acc)):
      best_valid_acc = train_acc.result().numpy()
      print("Saving net with valid_acc = ", best_valid_acc)
      net.save_weights(filepath = model_save_dir + "curr_best_model", save_format = 'tf')

    train_loss.reset_states()
    valid_loss.reset_states()
    train_acc.reset_states()
    valid_acc.reset_states()
  
  print("Saving final net")
  net.save_weights(filepath = model_save_dir + "training_end", save_format = 'tf')

def main(argv):
  data_dir = "./dataset/data"
  model_save_dir = "./saved_model"

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
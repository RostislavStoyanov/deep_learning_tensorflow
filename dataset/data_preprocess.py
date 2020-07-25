import sys, getopt, os, pickle
import numpy as np
import tensorflow as tf
import librosa

sys.path.insert(0,'..')
from shared_variables import subdirectory_list, n_fft, n_mels, t, hop_length, sr, genres


help_msg = 'data_preprocessing.py -d <dir>, where dir is the directory containing labeled sub-dirs in which the .wav files are located'

# The following functions can be used to convert a value to a type compatible
# with tf.Example. https://www.tensorflow.org/tutorials/load_data/tfrecord#tfexample

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def print_and_exit(msg,exit_code):
	print(msg)
	sys.exit(exit_code)

def add_padding(array):
	padded = np.zeros((sr * 10))
	padded[0:array.shape[0]] = array

	return padded

#https://www.tensorflow.org/tutorials/load_data/tfrecord#creating_a_tfexample_message
def spectogram_to_example(spectogram, genre):
	feature = {
		'label': _int64_feature(genre),
		'spectogram': _bytes_feature(spectogram)
	}

	return tf.train.Example(features = tf.train.Features(feature = feature))


def create_spectogram(dir_to_browse, writer, genre):
	files = os.listdir(dir_to_browse)

	for file in files:
		y, _ = librosa.load(dir_to_browse + '/' + file, sr = sr)
		if(y.shape[0] < (10 * sr)):
			y = add_padding(y)
		spectogram = librosa.feature.melspectrogram(y = y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
		spectogram_pwr = librosa.power_to_db(spectogram, ref = np.max)

		spectogram_1d = np.reshape(spectogram_pwr, (n_mels * t)).tobytes()
		example = spectogram_to_example(spectogram_1d, genre)
		writer.write(example.SerializeToString())



#what if we already have the pickle but have updated the directory with the files -- this wont work correctly
# better to first create the pickle and then compare with old one if such exists
def maybe_create_spectograms(main_dir):
	genre_map = dict((label, idx) for idx, label in enumerate(genres))

	for subdir in subdirectory_list:
		curr_tf_record_path = main_dir + '/' + subdir + '.tfrecord'
		curr_subdir_path = main_dir + '/' + subdir
		if os.path.isfile(curr_tf_record_path):
			print(curr_tf_record_path, "already exists... skipping")
			continue

		print("Writing to ", curr_tf_record_path, "... ", end = "", flush = True)
		writer = tf.io.TFRecordWriter(path = curr_tf_record_path)
		for genre in genres:
			dir_to_browse = curr_subdir_path + '/' + genre
			create_spectogram(dir_to_browse, writer, genre_map[genre])
		print("done")

def main(argv):
	main_dir = ""

	try:
		opts, _ = getopt.getopt(argv,"h:d:",["main_dir="])
	except:
		print_and_exit(help_msg, 2)

	for opt, arg in opts:
		if opt == '-h':
			print_and_exit(help_msg, 0)
		elif opt == '-d':
			main_dir = arg

	if main_dir == "" :
		print_and_exit(help_msg, 2)
	
	maybe_create_spectograms(main_dir)

if __name__ == "__main__":
	main(sys.argv[1:])
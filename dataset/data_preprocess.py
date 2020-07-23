import sys, getopt, os, pickle
import numpy as np
import librosa
#from multiprocessing import Pool

n_fft = 2048
hop_length = 512
n_mels = 128
t = 431
sr = 22050


help_msg = 'data_preprocessing.py -d <dir>, where dir is the directory containing labeled sub-dirs in which the .wav files are located'
genres = ['Pop music', "Rock music", 'Hip hop music', 'Techno', 'Rhythm and blues', 'Vocal music', 'Reggae']

def print_and_exit(msg,exit_code):
	print(msg)
	sys.exit(exit_code)

def add_padding(array):
	padded = np.zeros((sr * 10))
	padded[0:array.shape[0]] = array

	return padded

def create_spectogram(dir_to_browse, curr_pickle):
	files = os.listdir(dir_to_browse)
	data = np.ndarray(shape=(len(files), n_mels, t),
                         dtype=np.float32)
	data_idx = 0

	for file in files:
		y, _ = librosa.load(dir_to_browse + '/' + file, sr = sr)
		if(y.shape[0] < (10 * sr)):
			y = add_padding(y)
		spectogram = librosa.feature.melspectrogram(y = y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
		spectogram_pwr = librosa.power_to_db(spectogram, ref = np.max)

		data[data_idx, :, :] = spectogram_pwr 
		data_idx = data_idx + 1

	try:
		with open(curr_pickle, 'wb') as f:
			print("Attempting to save", curr_pickle, end = "...")
			pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
			print("done")
	except Exception as e:
		print('Unable to save data to', curr_pickle, ':', e)

#what if we already have the pickle but have updated the directory with the files -- this wont work correctly
# better to first create the pickle and then compare with old one if such exists
def maybe_create_spectograms(main_dir):
	for genre in genres:
		curr_pickle = main_dir + '/' + genre + ".pickle"
		if os.path.isfile(curr_pickle):
			print("File",curr_pickle,"already exists... skipping")
			continue

		dir_to_browse = main_dir + '/' + genre
		#pool = Pool()
		#pool.map(create_spectogram, dir_to_browse, curr_pickle)
		create_spectogram(dir_to_browse, curr_pickle)

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
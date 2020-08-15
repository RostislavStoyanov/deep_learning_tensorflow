genres = ['Pop music', "Rock music", 'Hip hop music', 'Techno', 'Rhythm and blues', 'Vocal music', 'Reggae']
subdirectory_list = ['train', 'valid', 'eval']

n_fft = 2048
hop_length = 512
n_mels = 128
t = 431
sr = 22050

BATCH_SIZE = 50
EPOCHS = 75
LEARNING_RATE = 0.035
#LR_DECAY = 0.94
EPSILON = 1.0

NUM_CLASSES = 7

CNN_1D_LAYERS_1TO4_FILTERS = 128
CNN_1D_LAYERS_1TO4_KERNEL_SIZE = 4
CNN_1D_LAYERS_1TO3_MAX_POOL_SIZE = 4
CNN_1D_LAYERS_4_MAX_POOL_SIZE = 2
CNN_1D_LAYER_5_FILTERS = 256
CNN_1D_LAYER_5_KERNEL_SIZE = 1
CNN_1D_LAYER_5_MAX_POOL_SIZE = 1
CNN_1D_LAYERS_DENSE_UNITS = 72

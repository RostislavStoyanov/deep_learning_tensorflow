import tensorflow as tf
from tensorflow import keras
import sys

sys.path.insert(0,'..')
from global_vars import NUM_CLASSES, CNN_1D_LAYER_5_FILTERS, CNN_1D_LAYER_5_KERNEL_SIZE, CNN_1D_LAYER_5_MAX_POOL_SIZE, CNN_1D_LAYERS_1TO4_FILTERS, CNN_1D_LAYERS_1TO4_KERNEL_SIZE, CNN_1D_LAYERS_1TO3_MAX_POOL_SIZE, CNN_1D_LAYERS_DENSE_UNITS, CNN_1D_LAYERS_4_MAX_POOL_SIZE, n_mels, t

class Conv1D_BatchNorm(keras.layers.Layer):
  def __init__(self, filters, kernel_size, strides = 1, padding = 'same'):
    super(Conv1D_BatchNorm, self).__init__()

    self.conv1d = keras.layers.Conv1D(filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.001), bias_regularizer = keras.regularizers.l2(0.001))
    self.batch_norm = keras.layers.BatchNormalization()
    #self.relu = keras.layers.ReLU()

  def call(self, inputs, training = None):
    conv = self.conv1d(inputs, training = training)
    conv_batch = self.batch_norm(conv, training = training)
    #relu = self.relu(conv_batch)

    return conv_batch


class CNN_1D(keras.Model):
  def __init__(self):
    super(CNN_1D, self).__init__()

    self.conv1 = Conv1D_BatchNorm(filters = CNN_1D_LAYERS_1TO4_FILTERS, kernel_size = CNN_1D_LAYERS_1TO4_KERNEL_SIZE)
    self.maxpool1 = keras.layers.MaxPool1D(pool_size = CNN_1D_LAYERS_1TO3_MAX_POOL_SIZE)
    
    self.conv2 = Conv1D_BatchNorm(filters = CNN_1D_LAYERS_1TO4_FILTERS, kernel_size = CNN_1D_LAYERS_1TO4_KERNEL_SIZE)
    self.maxpool2 = keras.layers.MaxPool1D(pool_size = CNN_1D_LAYERS_1TO3_MAX_POOL_SIZE)
    
    self.conv3 = Conv1D_BatchNorm(filters = CNN_1D_LAYERS_1TO4_FILTERS, kernel_size = CNN_1D_LAYERS_1TO4_KERNEL_SIZE)
    self.maxpool3 = keras.layers.MaxPool1D(pool_size = CNN_1D_LAYERS_1TO3_MAX_POOL_SIZE)
    
    self.conv4 = Conv1D_BatchNorm(filters = CNN_1D_LAYERS_1TO4_FILTERS, kernel_size = CNN_1D_LAYERS_1TO4_KERNEL_SIZE)
    self.maxpool4 = keras.layers.MaxPool1D(CNN_1D_LAYERS_4_MAX_POOL_SIZE)
    
    self.conv5 = Conv1D_BatchNorm(filters = CNN_1D_LAYER_5_FILTERS, kernel_size = CNN_1D_LAYER_5_KERNEL_SIZE)
    self.maxpool5 = keras.layers.MaxPool1D(pool_size = CNN_1D_LAYER_5_MAX_POOL_SIZE)
    
    self.flat = keras.layers.Flatten()
    self.dense1 = keras.layers.Dense(units = CNN_1D_LAYERS_DENSE_UNITS, kernel_regularizer = keras.regularizers.l2(0.001), bias_regularizer = keras.regularizers.l2(0.001))
    self.dropout1 = keras.layers.Dropout(rate = 0.5)
    self.softmax = keras.layers.Dense(units = NUM_CLASSES, activation = keras.activations.softmax)

  def model(self):
    x = keras.Input(shape = (n_mels, t))
    return keras.Model(inputs=[x], outputs = self.call(x))

  def call(self, inputs, training = False):
    layer1_conv = self.conv1(inputs, training = training)
    layer1_maxpool = self.maxpool1(layer1_conv, training = training)
    
    layer2_conv = self.conv2(layer1_maxpool, training = training)
    layer2_maxpool = self.maxpool2(layer2_conv)

    layer3_conv = self.conv3(layer2_maxpool, training = training)
    layer3_maxpool = self.maxpool3(layer3_conv)

    layer4_conv = self.conv4(layer3_maxpool, training = training)
    layer4_maxpool = self.maxpool4(layer4_conv)

    layer5_conv = self.conv5(layer4_maxpool, training = training)
    layer5_maxpool = self.maxpool5(layer5_conv)

    pre_dense_flat = self.flat(layer5_maxpool)
    dense = self.dense1(pre_dense_flat, training = training)
    drop = self.dropout1(dense)

    return self.softmax(drop)
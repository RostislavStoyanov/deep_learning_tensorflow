import tensorflow as tf
from tensorflow import keras
import sys

sys.path.insert(0,'..')
from global_vars import n_mels, t, BATCH_SIZE

class CNN(keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = keras.layers.Conv2D(64, kernel_size = (3,3), strides = 1,
                    padding = 'same', activation = 'relu')
        self.bn1 = keras.layers.BatchNormalization()
        self.avg_pool1 = keras.layers.MaxPool2D(pool_size = (2, 2))
        self.conv2 = keras.layers.Conv2D(64, kernel_size = (3,5), strides = 1,
                    padding = 'same', activation = 'relu')
        self.bn2 = keras.layers.BatchNormalization()
        self.avg_pool2 = keras.layers.MaxPool2D(pool_size = (2,2))
        self.conv3 = keras.layers.Conv2D(64, kernel_size = (3,5), strides = 1,
                    padding = 'same', activation = 'relu')
        self.bn3 = keras.layers.BatchNormalization()
        self.avg_pool3 = keras.layers.MaxPool2D(pool_size=(2,2))
        self.flatten = keras.layers.Flatten()
        self.dropout = keras.layers.Dropout(rate = 0.25)
        self.dense = keras.layers.Dense(units = 32)
        self.softmax = keras.layers.Dense(units = 7, activation = keras.activations.softmax)
    
    def call(self, inputs, training = None):
        conv = self.conv1(inputs, training = training)
        bn = self.bn1(conv, training = training)
        avg_pool = self.avg_pool1(bn)

        conv = self.conv2(avg_pool, training = training)
        bn = self.bn2(conv, training = training)
        avg_pool = self.avg_pool2(bn)

        conv = self.conv3(avg_pool, training = training)
        bn = self.bn3(conv, training = training)
        avg_pool = self.avg_pool3(bn)

        dropout = self.dropout(avg_pool)
        flat = self.flatten(dropout)
        dense = self.dense(flat, training = training)
        return self.softmax(dense)
    
    def model(self):
        x = keras.Input(shape = (n_mels, t, 1))
        return keras.Model(inputs=[x], outputs = self.call(x))

import tensorflow as tf
from tensorflow import keras
from .blocks import ResidualBlock, Conv2D_BatchNorm

class ResNet(tf.keras.Model):
  def __init__(self):
    super(ResNet, self).__init__()

    self.conv1 = Conv2D_BatchNorm(filters = 64, kernel_size = (7, 7),
                                  strides = 2, padding = 'same', activation = None)
    self.pool1 = keras.layers.MaxPool2D(pool_size = (3, 3), strides = 2, padding = 'same')

    self.layer1 = self.__stack_residual_layers(filter_cnt = 64, block_cnt = 3, stride = 1)
    self.layer2 = self.__stack_residual_layers(filter_cnt = 128, block_cnt = 4, stride = 2)
    self.layer3 = self.__stack_residual_layers(filter_cnt = 256, block_cnt = 6, stride = 2)
    self.layer4 = self.__stack_residual_layers(filter_cnt = 512, block_cnt = 3, stride = 2)

    self.avgpool = keras.layers.GlobalAveragePooling2D()
    self.dense = keras.layers.Dense(units = 7, activation = keras.activations.softmax)

  def __stack_residual_layers(self, filter_cnt, block_cnt, stride):
    block = keras.Sequential()
    
    for _ in range(block_cnt):
      block.add(ResidualBlock(filter_cnt = filter_cnt, stride = stride))
    
    return block

  def call(self, inputs, training = None):
    output = self.conv1(inputs, training)
    output = tf.nn.relu(output)
    output = self.pool1(output)
    output = self.layer1(output, training)
    output = self.layer2(output, training)
    output = self.layer3(output, training)
    output = self.layer4(output, training)
    output = self.avgpool(output)
    output = self.dense(output)
    
    return output
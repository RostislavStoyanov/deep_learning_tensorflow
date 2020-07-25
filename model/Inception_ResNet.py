import tensorflow as tf
from tensorflow import keras
from blocks import Stem, Inception_ResNet_A, Inception_ResNet_B, Inception_ResNet_C, ReductionA, ReductionB


class Inception_ResNet(tf.keras.Model):
  def __init__(self):
    super(Inception_ResNet, self).__init__()

    #set private variables
    self.resnet_A_count = 5
    self.resnet_B_count = 10
    self.resnet_C_count = 5

    #vars containing the main body of the net
    self.stem = Stem()
    self.inception_resnet_A = self.__stack_inception_resnet_layers(self.resnet_A_count, Inception_ResNet_A)
    self.reduction_A = ReductionA()
    self.inception_resnet_B = self.__stack_inception_resnet_layers(self.resnet_B_count, Inception_ResNet_B)
    self.reductionB = ReductionB
    self.inception_resnetC = self.__stack_inception_resnet_layers(self.resnet_C_count, Inception_ResNet_C)
    self.avg_pool = keras.layers.AveragePooling2D(pool_size = (8,8))
    self.dropout = keras.layers.Dropout(rate = 0.2)
    self.flat = keras.layers.Flatten()
    self.softmax = keras.layers.Dense(units = 7, activation = keras.activations.softmax) 
  
#stack multiple resnet_A blocks
  def __stack_inception_resnet_layers(self, count, layer):
    layers = keras.Sequential()
    
    for _ in range(count):
      layers.add(layer())
    
    return layers

def call(self, inputs):
  output = self.stem(inputs)
  output = self.inception_resnet_A(output)
  output = self.reduction_A(output)
  output = self.inception_resnet_B(output)
  output = self.reductionB(output)
  output = self.Inception_ResNet_C(output)
  output = self.avg_pool(output)
  output = self.dropout(output)
  output = self.flat(output)
  output = self.softmax(output)

  return output
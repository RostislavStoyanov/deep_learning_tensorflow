import tensorflow as tf
from tensorflow import keras

class Conv2D_BatchNorm(keras.layers.Layer):
  def __init__(self, filters, kernel_size, strides, padding, activation, kernel_regularizer = None):
    super(Conv2D_BatchNorm, self)
    
    # same parameter ordering as
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
    self.conv2d = keras.layers.Conv2D(filters = filters, kernel_size = kernel_size,
                strides = strides, padding = padding, activation = activation,
                 kernel_regularizer = kernel_regularizer)
    self.batch_norm = keras.layers.BatchNormalization()

    def call(self, inputs):
      x = self.conv(inputs)
      x = self.batch_norm(x, training = True)


class Stem(keras.layers.Layer):
  def __init__(self):
    super(Stem, self).__init__()

    #root
    self.conv1 = Conv2D_BatchNorm(filters = 32, kernel_size = (3,3), 
                strides = 2, padding = 'valid', activation = 'relu')
    self.conv2 = Conv2D_BatchNorm(filters = 32, kernel_size = (3,3),
                strides = 2, padding = 'valid', activation = 'relu')
    self.conv3 = Conv2D_BatchNorm(filters = 64, kernel_size = (3,3),
                strides = 1, padding = 'same', activation = 'relu')
    
    # first branching
    self.br1_maxpool = keras.layers.MaxPool2D(pool_size = (3,3), strides = 2,
                padding = 'valid')
    self.br2_conv = Conv2D_BatchNorm(filters = 96, kernel_size = (3,3),
                  strides = 2, padding = 'valid', activation = 'relu')
     
    #second branching 
    self.br3_conv1 = Conv2D_BatchNorm(filters = 64, kernel_size = (1,1),
                    strides = 1, padding = 'same', activation = 'relu')
    self.br3_conv2 = Conv2D_BatchNorm(filters = 96, kernel_size = (3,3),
                    strides = 1, padding = 'valid', activation = 'relu')
    self.br4_conv1 = Conv2D_BatchNorm(filters = 64, kernel_size = (1,1),
                strides = 1, padding = 'same', activation = 'relu')
    self.br4_conv2 = Conv2D_BatchNorm(filters = 64, kernel_size = (7,1),
                strides = 1, padding = 'same', activation = 'relu')
    self.br4_conv3 = Conv2D_BatchNorm(filters = 64, kernel_size = (1,7),
                strides = 1, padding = 'same', activation = 'relu')
    self.br4_conv4 = Conv2D_BatchNorm(filters = 96, kernel_size = (3,3),
                strides = 1, padding = 'valid', activation = 'relu')          
    #third branching 
    self.br5_conv = Conv2D_BatchNorm(filters = 192, kernel_size = (3,3),
                    strides = 1, padding = 'valid', activation = 'relu')
    self.br6_maxpool = keras.layers.MaxPool2D(pool_size = (3,3), strides = 2,
                padding = 'valid')

  def call(self, inputs):
    root = self.conv1.call(inputs)
    root = self.conv2.call(root)
    root = self.conv3.call(root)
    
    br_1 = self.br1_maxpool.call(root)
    br_2 = self.br2_conv.call(root)
    br_1_2 = tf.concat([br_1, br_2], -1, name = "stem_br_1_2_concat")
    
    br_3 = self.br3_conv1.call(br_1_2)
    br_3 = self.br3_conv2(br_3)
    br_4 = self.br4_conv1(br_1_2)
    br_4 = self.br4_conv2(br_4)
    br_4 = self.br4_conv3(br_4)
    br_4 = self.br4_conv4(br_4)
    br_3_4 = tf.concat([br_3, br_4], -1, name = "stem_br_3_4_concat")

    br5 = self.br5_conv(br_3_4)
    br6 = self.br6_maxpool(br_3_4)

    return tf.concat([br5, br6], -1, name = "stem_br_5_6_concat")

class Inception_ResNet_A(keras.layers.Layer):
  def __init__(self):
    super(Inception_ResNet_A, self).__init__()
    
    self.br1_conv = Conv2D_BatchNorm(filter = 32, kernel_size = (1,1),
                    strides = 1, padding = 'same', activation = 'relu')
    
    self.br2_conv1 = Conv2D_BatchNorm(filter = 32, kernel_size = (1,1),
                    strides = 1, padding = 'same', activation = 'relu')
    self.br2_conv2 = Conv2D_BatchNorm(filter = 32, kernel_size = (3,3),
                    strides = 1, padding = 'same', activation = 'relu')
    
    self.br3_conv1 = Conv2D_BatchNorm(filter = 32, kernel_size = (1,1),
                    strides = 1, padding = 'same', activation = 'relu')
    self.br3_conv2 = Conv2D_BatchNorm(filter = 48, kernel_size = (3,3),
                    strides = 1, padding = 'same', activation = 'relu')
    self.br3_conv3 = Conv2D_BatchNorm(filter = 64, kernel_size = (3,3),
                    strides = 1, padding = 'same', activation = 'relu')
    self.concat_conv = Conv2D_BatchNorm(filter = 384, kernel_size = (1,1),
                    strides = 1, padding = 'same', activation = None)

  def call(self, inputs):
    br1 = self.br1_conv.call(inputs)

    br2 = self.br2_conv1.call(inputs)
    br2 = self.br2_conv2.call(br2)

    br3 = self.br3_conv1.call(inputs)
    br3 = self.br3_conv2.call(br3)
    br3 = self.br3_conv3.call(br3)

    concat = tf.concat([br1, br2, br3], -1, name = "incept_res_A_concat")
    concat = self.concat_conv.call(concat)
    
    summation = keras.layers.add([concat, inputs]) * 0.1
    return tf.nn.relu(summation)

class Inception_ResNet_B(keras.layers.Layer):
  def __init__(self):
    super(Inception_ResNet_B, self).__init__()
    self.br1_conv = Conv2D_BatchNorm(filter = 192, kernel_size = (1,1),
                    strides = 1, padding = 'same', activation = 'relu')
    
    self.br2_conv1 = Conv2D_BatchNorm(filter = 128, kernel_size = (1,1),
                    strides = 1, padding = 'same', activation = 'relu')
    self.br2_conv2 = Conv2D_BatchNorm(filter = 160, kernel_size = (1,7),
                    strides = 1, padding = 'same', activation = 'relu')
    self.br2_conv3 = Conv2D_BatchNorm(filter = 192, kernel_size = (7,1),
                    strides = 1, padding = 'same', activation = 'relu')
    
    self.concat_conv = Conv2D_BatchNorm(filter = 1154, kernel_size = (1,1),
                    strides = 1, padding = 'same', activation = None)
    
    def call(self, inputs):
      br1 = self.br1_conv.call(inputs)

      br2 = self.br2_conv1.call(inputs)
      br2 = self.br2_conv2.call(br2)
      br2 = self.br2_conv3.call(br2)

      concat = tf.concat([br1, br2], -1, name = "incept_res_B_concat")
      concat = self.concat_conv.call(concat)

      summation = keras.layers.add([concat, inputs]) * 0.1
      return tf.nn.relu(summation)

class Inception_ResNet_C(keras.layers.Layer):
  def __init__(self):
    super(Inception_ResNet_C, self).__init__()
    
    self.br1_conv = Conv2D_BatchNorm(filter = 192, kernel_size = (1,1),
                    strides = 1, padding = 'same', activation = 'relu')
    
    self.br2_conv1 = Conv2D_BatchNorm(filter = 192, kernel_size = (1,1),
                    strides = 1, padding = 'same', activation = 'relu')
    self.br2_conv2 = Conv2D_BatchNorm(filter = 224, kernel_size = (1,3),
                    strides = 1, padding = 'same', activation = 'relu')
    self.br2_conv3 = Conv2D_BatchNorm(filter = 256, kernel_size = (3,1),
                    strides = 1, padding = 'same', activation = 'relu')
    
    self.concat_conv = Conv2D_BatchNorm(filter = 2048, kernel_size = (1,1),
                    strides = 1, padding = 'same', activation = None)
    
  def call(self, inputs):
    br1 = self.br1_conv.call(inputs)

    br2 = self.br2_conv1.call(inputs)
    br2 = self.br2_conv2.call(br2)
    br2 = self.br2_conv3.call(br2)

    concat = tf.concat([br1, br2], -1, name = "incept_res_C_concat")
    concat = self.concat_conv.call(concat)

    summation = keras.layers.add([concat, inputs]) * 0.1
    return tf.nn.relu(summation)
    

class ReductionA(keras.layers.Layer):
  def __init__(self):
    super(ReductionA, self).__init__()

    self.br1_maxpool = keras.layers.MaxPool2D(pool_size = (3,3), strides = 2,
                      padding = 'valid')
    #batch norm?

    self.br2_conv = Conv2D_BatchNorm(filter = 384, kernel_size = (3,3),
                    strides = 2, padding = 'valid', activation = 'relu')

    self.br3_conv1 = Conv2D_BatchNorm(filter = 256, kernel_size = (1,1),
                    strides = 1, padding = 'same', activation = 'relu')
    self.br3_conv2 = Conv2D_BatchNorm(filter = 256, kernel_size = (3,3),
                    strides = 1, padding = 'same', activation = 'relu')
    self.br3_conv3 = Conv2D_BatchNorm(filter = 384, kernel_size = (3,3),
                    strides = 2, padding = 'valid', activation = 'relu')
      
  def call(self, inputs):
    br1 = self.br1_maxpool.call(inputs)
    
    br2 = self.br2_conv.call(inputs)

    br3 = self.br3_conv1.call(inputs)
    br3 = self.br3_conv2.call(br3)
    br3 = self.br3_conv3.call(br3)

    return tf.concat([br1, br2, br3], -1, name = "reductionA_concat")

class ReductionB(keras.layers.Layer):
  def __init__(self):
    super(ReductionB, self).__init__()

    self.br1_maxpool = keras.layers.MaxPool2D(pool_size = (3,3), strides = 2,
                      padding = 'valid')
    
    self.br2_conv1 = Conv2D_BatchNorm(filter = 256, kernel_size = (1,1),
                    strides = 1, padding = 'same', activation = 'relu')
    self.br2_conv2 = Conv2D_BatchNorm(filter = 384, kernel_size = (3,3),
                    strides = 2, padding = 'valid', activation = 'relu')
    
    self.br3_conv1 = Conv2D_BatchNorm(filter = 256, kernel_size = (1,1),
                    strides = 1, padding = 'same', activation = 'relu')
    self.br3_conv2 = Conv2D_BatchNorm(filter = 288, kernel_size = (3,3),
                    strides = 2, padding = 'valid', activation = 'relu')
    
    self.br4_conv1 = Conv2D_BatchNorm(filter = 256, kernel_size = (1,1),
                    strides = 1, padding = 'same', activation = 'relu')
    self.br4_conv2 = Conv2D_BatchNorm(filter = 288, kernel_size = (3,3),
                    strides = 1, padding = 'same', activation = 'relu')
    self.br4_conv3 = Conv2D_BatchNorm(filter = 320, kernel_size = (3,3),
                    strides = 2, padding = 'valid', activation = 'relu')
  
  def call(self, inputs):
    br1 = self.br1_maxpool.call(inputs)

    br2 = self.br2_conv1.call(inputs)
    br2 = self.br2_conv2.call(br2)
    
    br3 = self.br3_conv1.call(inputs)
    br3 = self.br3_conv2.call(br3)

    br4 = self.br4_conv1.call(inputs)
    br4 = self.br4_conv2.call(br4)
    br4 = self.br4_conv3.call(br4)

    return tf.concat([br1, br2, br3, br4], -1, name = "reductionB_concat")
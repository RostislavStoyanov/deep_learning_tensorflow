# deep_learning_tensorflow
Repository containing course project for deep learning with tensorflow course in FMI

### General information 
The purpose of this project was to check if writing deeper CNN using residual connections for example helps inidentifing music genres. Mostly following [Music Genre Classification using Machine Learning Techniques](https://arxiv.org/pdf/1804.01149.pdf). CNN code based on [Inception-v4, Inception-ResNet andthe Impact of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261.pdf)

### Results 
Training done for 25 epochs using SGD with learning_rate = 0.05 and batch_size of 16 for 2d networks.
| Architecture       | Best accuracy achieved  
| :------------- | :----------: |
|  Basic 2D CNN | 33.07%   |
| ResNet   | 60.08% |
| Inception-ResNet | 60.48% 
| 1D-CNN | 53.47$

### Future work
* add preprocessing
* write deeper 1-d cnn
* use different dataset?

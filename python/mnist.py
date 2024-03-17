import numpy as np
from keras.datasets import mnist

# set numpy random seed for reproducibility
np.random.seed(42)

# load mnist data from keras datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape x_train from 60000x28x28 to 60000x784
x_train = x_train.reshape(60000, 784)

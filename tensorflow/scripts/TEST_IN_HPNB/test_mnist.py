from keras.datasets import mnist
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("ndim= ", train_images.ndim)
print("shape= ", train_images.shape)
print("dtype= ", train_images.dtype)

# Display the fourth number
digit = train_images[4]

import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

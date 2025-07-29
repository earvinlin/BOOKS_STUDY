"""
§1.6.2 Sequential Model
"""
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
from keras import layers

#from tensorflow.keras.utils import plot_model
from keras.utils import plot_model
from IPython.display import Image

model = keras.Sequential(name='Sequential')

"""
# [ Method 1 ]
# 202506011 tensorflow 2.18.1 Input需要改寫如下方式
#model.add(layers.Dense(64, activation='relu', input_shape=(784,)))
model.add(layers.Input(shape=(784,)))

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])
"""
        
# [ Method 2 ]
model = tf.keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')])

plot_model(model, to_file='Functional_API_Sequential_Model.png')

Image('Functional_API_Sequential_Model.png')

"""
#img = Image.open('Functional_API_Sequential_model.png')
from PIL import Image
from IPython.display import display
img = Image.open('Functional_API_Sequential_model.png')
display(img)
"""

"""
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
os.chdir('/home/earvin/workspaces/GithubProjects/tensorflow/scripts/BOOK_EasyStudyTF2')
image1 = img.imread('Functional_API_Sequential_model.png')
plt.imshow(image1)
plt.show()
"""

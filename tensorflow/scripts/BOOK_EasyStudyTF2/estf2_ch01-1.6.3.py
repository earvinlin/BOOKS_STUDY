"""
§1.6.3 Function API
"""
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.utils import plot_model
from IPython.display import Image

#################################
# Single Input and Output Model #
#################################
inputs = keras.Input(shape=(784,), name='Input')
h1 = layers.Dense(64, activation='relu', name='hidden1')(inputs)
h2 = layers.Dense(64, activation='relu', name='hidden2')(h1)
outputs = layers.Dense(10, activation='softmax', name='Output')(h2)

model = keras.Model(inputs=inputs, outputs=outputs)

plot_model(model, to_file='Functional_API_Single_Input_And_Output_Model.png')
Image('Functional_API_Single_Input_And_Output_Model.png')


#####################
# Multi Input Model #
#####################
# 網路模型輸入層
img_input = keras.Input(shape=(128, 128, 3), name='Image_Input')
info_input = keras.Input(shape=(1,), name='Information_Input')

# 網路模型隱藏層
h1_1 = layers.Conv2D(64, 5, strides=2, activation='relu', 
                     name='hidden1_1')(img_input)
h1_2 = layers.Conv2D(64, 5, strides=2, activation='relu', 
                     name='hidden1_2')(h1_1)
h1_2_ft = layers.Flatten()(h1_2)
h1_3 = layers.Dense(64, activation='relu', name='hidden1_3')(info_input)
concat = layers.Concatenate()([h1_2_ft, h1_3])
h2 = layers.Dense(64, activation='relu', name='hidden2')(concat)

# 網路模型輸出層
outputs = layers.Dense(1, name='Output')(h2)

# 建立網路輸出層
model = keras.Model(inputs=[img_input, info_input], outputs=outputs)

# 顯示網路模型架構
plot_model(model, to_file='Functional_API_Multi_Input_Model.png')
Image('Functional_API_Multi_Input_Model.png')

######################
# Multi Output Model #
######################
# 網路模型輸入層
inputs = keras.Input(shape=(28, 28, 1), name='Input')

# 網路模型隱藏層
h1 = layers.Conv2D(64, 3, activation='relu', name='hidden1')(inputs)
h2 = layers.Conv2D(64, 3, strides=2, activation='relu', name='hidden2')(h1)
h3 = layers.Conv2D(64, 3, strides=2, activation='relu', name='hidden3')(h2)
flatten = layers.Flatten()(h3)

# 網路模型輸出層
age_output = layers.Dense(1, name='Age_Output')(flatten)
gender_output = layers.Dense(1, name='Gender_Output')(flatten)

# 建立網路模型
model = keras.Model(inputs=inputs, outputs=[age_output, gender_output])

# 顯示網路模型架構
plot_model(model, to_file='Functional_API_Multi_Output_Model.png')
Image('Functional_API_Multi_Output_Model')


################################
# Multi Input and OutPut Model #
################################
# 網路模型輸入層
image_inputs = keras.Input(shape=(256, 256, 3), name='Image_Input')
info_inputs = keras.Input(shape=(10,), name='Info_input')

# 網路模型隱藏層(Image Input)
h1 = layers.Conv2D(64, 3, activation='relu', name='hidden1')(image_inputs)
h2 = layers.Conv2D(64, 3, strides=2, activation='relu', name='hidden2')(h1)
h3 = layers.Conv2D(64, 3, strides=2, activation='relu', name='hidden3')(h2)
flatten = layers.Flatten()(h3)

# 網路模型隱藏層(Information Input)
h4 = layers.Dense(64)(info_inputs)
concat = layers.Concatenate()([flatten, h4]) # 結合Image and Information特徵

# 網路模型輸出層
weather_outputs = layers.Dense(1, name='Output1')(concat)
temp_outputs = layers.Dense(1, name='Output2')(concat)
humidity_outputs = layers.Dense(1, name='Output3')(concat)

# 建立網路模型
model = keras.Model(inputs=[image_inputs, info_inputs],
                    outputs=[weather_outputs, temp_outputs, humidity_outputs])

# 顯示網路模型架構
plot_model(model, to_file='Functional_API_Multi_Input_And_Output_Model.png')
Image('Functional_API_Multi_Input_And_Output_Model.png')



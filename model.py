import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense,Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.model_selection import train_test_split
from Augmentation import dataset

data_dir='./dataset/'

# Training the deep learning model

input_img = Input(shape=(600,600,3))

x = Conv2D(16, 3, 1, activation='relu', padding='same')(input_img) #nb_filter, nb_row, nb_col
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, 3, 1, activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, 3, 1, activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)


x = Conv2D(8, 3, 1, activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(8, 3, 1, activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, 3, 1, activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

decoded = Conv2D(1, 5, 1, activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='RMSprop', loss='mean_squared_error')
#autoencoder.summary()


# Data acqusation
# Reading the files
# Image data generators setup
train_datagen = image.ImageDataGenerator()
im_train,im_test,m_train,m_test = dataset()
train_gen = train_datagen.flow(im_train,m_train,batch_size=32)

history = autoencoder.fit(train_gen,epochs=20,validation_data=(im_test,m_test))

plt.figure(figsize=(10,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.savefig('history.png')

# model save
autoencoder.save('model.h5')
#!/usr/bin/env python
# coding: utf-8

# # import important libraries
# 

# In[1]:


import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA


# In[2]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import initializers


# # creating labels for the dataset with digits 1,3,6,7,9

# In[3]:


def read_data(path):
    
    train_path = path+"/train"
    test_path = path+"/test"
    validation_path = path+"/val"
    
    tr_data, test_data, val_data = [], [], []
    tr_out, test_out, val_out = [], [], []

    for i in os.listdir(train_path):

        if i != ".DS_Store":
            for j in os.listdir(train_path+"/"+i):
                tr_data.append(cv2.imread(train_path+"/"+i+"/"+j, cv2.IMREAD_GRAYSCALE))
                tr_out.append(i)

            for j in os.listdir(test_path+"/"+i):
                test_data.append(cv2.imread(test_path+"/"+i+"/"+j, cv2.IMREAD_GRAYSCALE))
                test_out.append(i)

            for j in os.listdir(validation_path+"/"+i):
                val_data.append(cv2.imread(validation_path+"/"+i+"/"+j, cv2.IMREAD_GRAYSCALE))
                val_out.append(i)
                
                
    tr_data, test_data, val_data = np.array(tr_data), np.array(test_data), np.array(val_data)
    tr_out, test_out, val_out = np.array(list(map(int, tr_out))), np.array(list(map(int, test_out))), np.array(list(map(int, val_out)))

    return tr_data, test_data, val_data, tr_out, test_out, val_out


# In[4]:


def convert(data):
  out = []
  for i in data:
    if i == 1:
      out.append(0)
    if i == 3:
      out.append(1)
    if i == 6:
      out.append(2)
    if i == 7:
      out.append(3)
    if i == 9:
      out.append(4)

  return np.array(out)


# In[6]:


path = "C:/Users/Deepak Angra/Downloads/Group_4/Group_4"
train_data, test_data, val_data, train_out, test_out, val_out = read_data(path)


# In[7]:


train_label = convert(train_out)
val_label = convert(val_out)
test_label = convert(test_out)


# In[8]:


print(test_data.shape)


# In[9]:


# Flatten each image to a 784-dimensional vector
train_data = train_data.reshape((-1, 784))
val_data = val_data.reshape((-1, 784))
test_data = test_data.reshape((-1, 784))


# In[10]:


# Normalize the pixel values to [0, 1]
train_data = train_data / 255.0
val_data = val_data / 255.0
test_data = test_data / 255.0


# # Autoencoder 128 Hidden layer = 1

# In[22]:


initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)


# In[23]:


# Define input shape
inputs = Input(shape=(784,))

# Define encoding layer
encoded = Dense(128, activation='tanh',kernel_initializer=initializer)(inputs)

# Define decoding layer
decoded = Dense(784, activation='tanh',kernel_initializer=initializer)(encoded)

# Define autoencoder model
autoencoder_128 = Model(inputs=inputs, outputs=decoded)

# Compile the model
autoencoder_128.compile(optimizer='adam', loss='mse')
callback = EarlyStopping(monitor="val_loss",min_delta=0.0001, verbose = 0, restore_best_weights=True, patience = 5)
# Train the model
autoencoder_128.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=100000, batch_size=32,shuffle = True, callbacks = callback)

# Evaluate the model on the test set
test_loss1 = autoencoder_128.evaluate(test_data, test_data)
print('Average reconstruction error:', test_loss1)


# In[24]:


val_pred = autoencoder_128.predict(val_data)
val_reconstruction_error = keras.metrics.mean_squared_error(val_data, val_pred).numpy().mean()
print('Average reconstruction error on validation data:', val_reconstruction_error)


# In[25]:


train_loss1 = autoencoder_128.evaluate(train_data, train_data)
print('Average reconstruction error for train data:', train_loss1)


# In[26]:


# Select one image from each class in the training set
train_indices = [np.where(train_label==i)[0][0] for i in range(5)]
train_images = train_data[train_indices]

# Select one image from each class in the validation set
val_indices = [np.where(val_label==i)[0][0] for i in range(5)]
val_images = val_data[val_indices]

# Select one image from each class in the test set
test_indices = [np.where(test_label==i)[0][0] for i in range(5)]
test_images = test_data[test_indices]

# Reconstruct images for each architecture
reconstructed_images_1 = autoencoder_128.predict(test_images)
reconstructed_images_2 = autoencoder_128.predict(val_images)
reconstructed_images_3 = autoencoder_128.predict(train_images)

# Display original and reconstructed images for each class
fig, axs = plt.subplots(nrows=5, ncols=6, figsize=(15,16))

for i in range(5):
    axs[i][0].imshow(test_images[i].reshape(28,28), cmap='gray')
    axs[i][0].axis('off')
    
    axs[i][1].imshow(reconstructed_images_1[i].reshape(28,28), cmap='gray')
    axs[i][1].axis('off')
    
    axs[i][2].imshow(val_images[i].reshape(28,28), cmap='gray')
    axs[i][2].axis('off')
    
    axs[i][3].imshow(reconstructed_images_2[i].reshape(28,28), cmap='gray')
    axs[i][3].axis('off')
    
    axs[i][4].imshow(train_images[i].reshape(28,28), cmap='gray')
    axs[i][4].axis('off')
    
    axs[i][5].imshow(reconstructed_images_3[i].reshape(28,28), cmap='gray')
    axs[i][5].axis('off')
    
    if True:
        axs[i][0].set_title('Original Test Image')
        axs[i][1].set_title('Test_Image')
        axs[i][2].set_title('Original Val Image')
        axs[i][3].set_title('Val_Image')
        axs[i][4].set_title('Original Train Image')
        axs[i][5].set_title('Train_Image')
    
plt.tight_layout()
plt.show()


# # Autoencoder 128 Hidden layer = 3

# In[28]:


initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
# Define input shape
inputs = Input(shape=(784,))

# Define encoding layers
encoded = Dense(400, activation='tanh',kernel_initializer=initializer)(inputs)
encoded = Dense(128, activation='tanh',kernel_initializer=initializer)(encoded)

# Define decoding layers 
decoded = Dense(400, activation='tanh',kernel_initializer=initializer)(encoded)

# Define output layer 
decoded = Dense(784, activation='tanh',kernel_initializer=initializer)(decoded)

# Define autoencoder model
autoencoder_128_3 = Model(inputs=inputs, outputs=decoded)

# Compile the model
autoencoder_128_3.compile(optimizer='adam', loss='mse')
callback = EarlyStopping(monitor="val_loss",min_delta=0.0001, verbose = 0, restore_best_weights=True, patience = 5)
# Train the model
autoencoder_128_3.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=100000, batch_size=32,callbacks = callback, shuffle = True)



# Evaluate the model on the test set
test_loss2 = autoencoder_128_3.evaluate(test_data, test_data)
print('Average reconstruction error:', test_loss2)


# In[29]:


val_pred = autoencoder_128_3.predict(val_data)
val_reconstruction_error = keras.metrics.mean_squared_error(val_data, val_pred).numpy().mean()
print('Average reconstruction error on validation data:', val_reconstruction_error)


# In[31]:


train_loss1 = autoencoder_128_3.evaluate(train_data, train_data)
print('Average reconstruction error for train data:', train_loss1)


# In[32]:


# Select one image from each class in the training set
train_indices = [np.where(train_label==i)[0][0] for i in range(5)]
train_images = train_data[train_indices]

# Select one image from each class in the validation set
val_indices = [np.where(val_label==i)[0][0] for i in range(5)]
val_images = val_data[val_indices]

# Select one image from each class in the test set
test_indices = [np.where(test_label==i)[0][0] for i in range(5)]
test_images = test_data[test_indices]

# Reconstruct images for each architecture
reconstructed_images_1 = autoencoder_128.predict(test_images)
reconstructed_images_2 = autoencoder_128.predict(val_images)
reconstructed_images_3 = autoencoder_128.predict(train_images)

# Display original and reconstructed images for each class
fig, axs = plt.subplots(nrows=5, ncols=6, figsize=(15,16))

for i in range(5):
    axs[i][0].imshow(test_images[i].reshape(28,28), cmap='gray')
    axs[i][0].axis('off')
    
    axs[i][1].imshow(reconstructed_images_1[i].reshape(28,28), cmap='gray')
    axs[i][1].axis('off')
    
    axs[i][2].imshow(val_images[i].reshape(28,28), cmap='gray')
    axs[i][2].axis('off')
    
    axs[i][3].imshow(reconstructed_images_2[i].reshape(28,28), cmap='gray')
    axs[i][3].axis('off')
    
    axs[i][4].imshow(train_images[i].reshape(28,28), cmap='gray')
    axs[i][4].axis('off')
    
    axs[i][5].imshow(reconstructed_images_3[i].reshape(28,28), cmap='gray')
    axs[i][5].axis('off')
    
    if True:
        axs[i][0].set_title('Original Test Image')
        axs[i][1].set_title('Test_Image')
        axs[i][2].set_title('Original Val Image')
        axs[i][3].set_title('Val_Image')
        axs[i][4].set_title('Original Train Image')
        axs[i][5].set_title('Train_Image')
    
plt.tight_layout()
plt.show()


# In[ ]:





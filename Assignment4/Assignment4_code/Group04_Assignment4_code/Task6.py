#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA


# In[37]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, GaussianNoise
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import initializers


# In[38]:


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


# In[39]:


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


# In[40]:


path = "C:/Users/Deepak Angra/Downloads/Group_4/Group_4"
train_data, test_data, val_data, train_out, test_out, val_out = read_data(path)


# In[41]:


train_label = convert(train_out)
val_label = convert(val_out)
test_label = convert(test_out)


# In[42]:


# Flatten each image to a 784-dimensional vector
train_data = train_data.reshape((-1, 784))
val_data = val_data.reshape((-1, 784))
test_data = test_data.reshape((-1, 784))


# In[43]:


# Normalize the pixel values to [0, 1]
train_data = train_data / 255.0
val_data = val_data / 255.0
test_data = test_data / 255.0


# In[58]:


initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
# Define input shape
input_shape = (784,) 

# Define the encoder architecture
input_layer = Input(shape=input_shape)
encoded = Dense(units=32, activation='tanh')(input_layer)
#encoded = GaussianNoise(0.2)(encoded)  # Add 20% noise to the encoded layer

# Define the decoder architecture
decoded = Dense(units=784, activation='tanh')(encoded)

# Define the autoencoder as a model
autoencoder = Model(input_layer, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')
callback = EarlyStopping(monitor="val_loss",min_delta=0.0001, verbose = 0, restore_best_weights=True, patience = 5)
# # Train the model
autoencoder.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=100000, batch_size=32,shuffle = True, callbacks = callback)
# # Train the model
# autoencoder.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=10, batch_size=32, shuffle=True)


# In[59]:


# # Evaluate the model on the test,val,train set
test_loss1 = autoencoder.evaluate(test_data, test_data)
print('Average reconstruction error for test data:', test_loss1)
val_loss1 = autoencoder.evaluate(val_data, val_data)
print('Average reconstruction error for val data:', val_loss1)
train_loss1 = autoencoder.evaluate(train_data, train_data)
print('Average reconstruction errorfor train data:', train_loss1)


# In[64]:


# Get the weights between the input layer and the compressed layer
weights = autoencoder.layers[1].get_weights()
w1 = []

for w in weights[0].T:
    w1.append(w.reshape(28, 28))
    
w1 = np.array(w1)
w1.shape

plt.figure(figsize=(25, 25))
for i in range(16):
    ax = plt.subplot(4, 4, i+1)
    plt.title("Node no: "+str(i))
    plt.imshow(w1[i], cmap='binary')

plt.show()


# In[61]:


# Get the weights between the input layer and the compressed layer
weights = autoencoder.layers[1].get_weights()
w1 = []

for w in weights[0].T:
    w1.append(w.reshape(28, 28))
    
w1 = np.array(w1)
w1.shape

plt.figure(figsize=(25, 25))
l=16
for i in range(16):
    ax = plt.subplot(4, 4, i+1)
    plt.title("Node no: "+str(l+1))
    plt.imshow(w1[l], cmap='binary')
    l=l+1
plt.show()


# # Autoencoder noise 20%

# In[65]:


initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
# Define input shape
input_shape = (784,) 

# Define the encoder architecture
input_layer = Input(shape=input_shape)
encoded = Dense(units=32, activation='tanh')(input_layer)
encoded = GaussianNoise(0.2)(encoded)  # Add 20% noise to the encoded layer

# Define the decoder architecture
decoded = Dense(units=784, activation='tanh')(encoded)

# Define the autoencoder as a model
autoencoder = Model(input_layer, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')
callback = EarlyStopping(monitor="val_loss",min_delta=0.0001, verbose = 0, restore_best_weights=True, patience = 5)
# # Train the model
autoencoder.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=100000, batch_size=32,shuffle = True, callbacks = callback)
# # Train the model
# autoencoder.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=10, batch_size=32, shuffle=True)


# In[66]:


# # Evaluate the model on the test,val,train set
test_loss1 = autoencoder.evaluate(test_data, test_data)
print('Average reconstruction error for test data:', test_loss1)
val_loss1 = autoencoder.evaluate(val_data, val_data)
print('Average reconstruction error for val data:', val_loss1)
train_loss1 = autoencoder.evaluate(train_data, train_data)
print('Average reconstruction errorfor train data:', train_loss1)


# In[67]:


# Get the weights between the input layer and the compressed layer
weights = autoencoder.layers[1].get_weights()
w1 = []

for w in weights[0].T:
    w1.append(w.reshape(28, 28))
    
w1 = np.array(w1)
w1.shape

plt.figure(figsize=(25, 25))
for i in range(16):
    ax = plt.subplot(4, 4, i+1)
    plt.title("Node no: "+str(i))
    plt.imshow(w1[i], cmap='binary')

plt.show()


# In[68]:


# Get the weights between the input layer and the compressed layer
weights = autoencoder.layers[1].get_weights()
w1 = []

for w in weights[0].T:
    w1.append(w.reshape(28, 28))
    
w1 = np.array(w1)
w1.shape

plt.figure(figsize=(25, 25))
l=16
for i in range(16):
    ax = plt.subplot(4, 4, i+1)
    plt.title("Node no: "+str(l+1))
    plt.imshow(w1[l], cmap='binary')
    l=l+1
plt.show()


# # Autoencoder noise 40% 

# In[69]:


initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
# Define input shape
input_shape = (784,) 

# Define the encoder architecture
input_layer = Input(shape=input_shape)
encoded = Dense(units=32, activation='tanh')(input_layer)
encoded = GaussianNoise(0.4)(encoded)  # Add 20% noise to the encoded layer

# Define the decoder architecture
decoded = Dense(units=784, activation='tanh')(encoded)

# Define the autoencoder as a model
autoencoder = Model(input_layer, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')
callback = EarlyStopping(monitor="val_loss",min_delta=0.0001, verbose = 0, restore_best_weights=True, patience = 5)
# # Train the model
autoencoder.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=100000, batch_size=32,shuffle = True, callbacks = callback)
# # Train the model
# autoencoder.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=10, batch_size=32, shuffle=True)


# In[70]:


# # Evaluate the model on the test,val,train set
test_loss1 = autoencoder.evaluate(test_data, test_data)
print('Average reconstruction error for test data:', test_loss1)
val_loss1 = autoencoder.evaluate(val_data, val_data)
print('Average reconstruction error for val data:', val_loss1)
train_loss1 = autoencoder.evaluate(train_data, train_data)
print('Average reconstruction errorfor train data:', train_loss1)


# In[71]:


# Get the weights between the input layer and the compressed layer
weights = autoencoder.layers[1].get_weights()
w1 = []

for w in weights[0].T:
    w1.append(w.reshape(28, 28))
    
w1 = np.array(w1)
w1.shape

plt.figure(figsize=(25, 25))
for i in range(16):
    ax = plt.subplot(4, 4, i+1)
    plt.title("Node no: "+str(i))
    plt.imshow(w1[i], cmap='binary')

plt.show()


# In[74]:


# Get the weights between the input layer and the compressed layer
weights = autoencoder.layers[1].get_weights()
w1 = []

for w in weights[0].T:
    w1.append(w.reshape(28, 28))
    
w1 = np.array(w1)
w1.shape

plt.figure(figsize=(25, 25))
l=16
for i in range(16):
    ax = plt.subplot(4, 4, i+1)
    plt.title("Node no: "+str(l+1))
    plt.imshow(w1[l],cmap='binary')
    l=l+1
plt.show()


# In[ ]:





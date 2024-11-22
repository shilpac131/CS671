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
from tensorflow.keras.layers import Input, Dense, Flatten, GaussianNoise
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


# In[5]:


path = "C:/Users/shilp/OneDrive/Documents/CS671/Group_4/"
train_data, test_data, val_data, train_out, test_out, val_out = read_data(path)


# In[6]:


train_label = convert(train_out)
val_label = convert(val_out)
test_label = convert(test_out)


# In[7]:


# print(test_data.shape)


# In[8]:


# Flatten each image to a 784-dimensional vector
train_data = train_data.reshape((-1, 784))
val_data = val_data.reshape((-1, 784))
test_data = test_data.reshape((-1, 784))


# In[9]:


# Normalize the pixel values to [0, 1]
train_data = train_data / 255.0
val_data = val_data / 255.0
test_data = test_data / 255.0


# # Architecture 4: 512-128-64

# # Autoencoder 32 Hidden layer = 1, noise = 0.2

# In[24]:


initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
# Define input shape
input_shape = (784,) 

# Define the encoder architecture
input_layer = Input(shape=input_shape)
encoded = GaussianNoise(0.2)(input_layer)  # Add 20% noise to the encoded layer
encoded_32 = Dense(units=32, activation='tanh')(encoded)

# Define the decoder architecture
decoded = Dense(units=784, activation='tanh')(encoded_32)

# Define the autoencoder as a model
autoencoder = Model(input_layer, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')
callback = EarlyStopping(monitor="val_loss",min_delta=0.0001, verbose = 0, restore_best_weights=True, patience = 5)
# # Train the model
autoencoder.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=100000, batch_size=32,shuffle = True, callbacks = callback)
# # Train the model
# autoencoder.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=10, batch_size=32, shuffle=True)


# In[25]:


# # Evaluate the model on the test,val,train set
test_loss1 = autoencoder.evaluate(test_data, test_data)
print('Average reconstruction error for test data:', test_loss1)
val_loss1 = autoencoder.evaluate(val_data, val_data)
print('Average reconstruction error for val data:', val_loss1)
train_loss1 = autoencoder.evaluate(train_data, train_data)
print('Average reconstruction errorfor train data:', train_loss1)


# In[26]:


test_pred = autoencoder.predict(test_data)
test_reconstruction_error = keras.metrics.mean_squared_error(test_data, test_pred).numpy().mean()
print('Average reconstruction error on testing data:', test_reconstruction_error)
train_pred = autoencoder.predict(train_data)
train_reconstruction_error = keras.metrics.mean_squared_error(train_data, train_pred).numpy().mean()
print('Average reconstruction error on training data:', train_reconstruction_error)
val_pred = autoencoder.predict(val_data)
val_reconstruction_error = keras.metrics.mean_squared_error(val_data, val_pred).numpy().mean()
print('Average reconstruction error on validation data:', val_reconstruction_error)


# In[27]:


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
reconstructed_images_1 = autoencoder.predict(test_images)
reconstructed_images_2 = autoencoder.predict(val_images)
reconstructed_images_3 = autoencoder.predict(train_images)

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


# In[30]:


compressed_model_32 = Model(inputs=input_layer, outputs=encoded_32)


# In[31]:


# Use the compressed model to extract the compressed representation of the training data
compressed_32_train_data = compressed_model_32.predict(train_data)
compressed_32_test_data = compressed_model_32.predict(test_data)
compressed_32_val_data = compressed_model_32.predict(val_data)


# # FCNN Best Arch : 512-128-64

# In[32]:


initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Define input shape
inputs = Input(shape=(32,))

# Define hidden layers with tanh activation
x = Dense(512, activation='tanh',kernel_initializer=initializer)(inputs)
x = Dense(128, activation='tanh',kernel_initializer=initializer)(x)
x = Dense(64, activation='tanh',kernel_initializer=initializer)(x)
# Define output layer with softmax activation
outputs = Dense(5, activation='softmax')(x)

# Define model
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
callback = EarlyStopping(monitor="loss",min_delta=0.0001, verbose = 0, restore_best_weights=True, patience = 2)
# Train the model
model_fit = model.fit(compressed_32_train_data, train_label, validation_data=(compressed_32_val_data, val_label), epochs=10000, batch_size=32,callbacks=callback,shuffle = True)



# In[33]:


# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(compressed_32_val_data, val_label)
print('Validation accuracy:', val_acc)
test_loss, test_acc = model.evaluate(compressed_32_test_data, test_label)
print('Test accuracy:', test_acc)


# # Autoencoder 32 Hidden layer = 1, noise = 0.4

# In[35]:


initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
# Define input shape
input_shape = (784,) 

# Define the encoder architecture
input_layer = Input(shape=input_shape)
encoded = GaussianNoise(0.4)(input_layer)  # Add 40% noise to the encoded layer
encoded_32 = Dense(units=32, activation='tanh')(encoded)

# Define the decoder architecture
decoded = Dense(units=784, activation='tanh')(encoded_32)

# Define the autoencoder as a model
autoencoder = Model(input_layer, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')
callback = EarlyStopping(monitor="val_loss",min_delta=0.0001, verbose = 0, restore_best_weights=True, patience = 5)
# # Train the model
autoencoder.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=100000, batch_size=32,shuffle = True, callbacks = callback)
# # Train the model
# autoencoder.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=10, batch_size=32, shuffle=True)


# In[36]:


# # Evaluate the model on the test,val,train set
test_loss1 = autoencoder.evaluate(test_data, test_data)
print('Average reconstruction error for test data:', test_loss1)
val_loss1 = autoencoder.evaluate(val_data, val_data)
print('Average reconstruction error for val data:', val_loss1)
train_loss1 = autoencoder.evaluate(train_data, train_data)
print('Average reconstruction errorfor train data:', train_loss1)


# In[37]:


test_pred = autoencoder.predict(test_data)
test_reconstruction_error = keras.metrics.mean_squared_error(test_data, test_pred).numpy().mean()
print('Average reconstruction error on testing data:', test_reconstruction_error)
train_pred = autoencoder.predict(train_data)
train_reconstruction_error = keras.metrics.mean_squared_error(train_data, train_pred).numpy().mean()
print('Average reconstruction error on training data:', train_reconstruction_error)
val_pred = autoencoder.predict(val_data)
val_reconstruction_error = keras.metrics.mean_squared_error(val_data, val_pred).numpy().mean()
print('Average reconstruction error on validation data:', val_reconstruction_error)


# In[38]:


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
reconstructed_images_1 = autoencoder.predict(test_images)
reconstructed_images_2 = autoencoder.predict(val_images)
reconstructed_images_3 = autoencoder.predict(train_images)

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


# # FCNN Best arch - 512-128-64

# In[40]:


compressed_model_32 = Model(inputs=input_layer, outputs=encoded_32)
# Use the compressed model to extract the compressed representation of the training data
compressed_32_train_data = compressed_model_32.predict(train_data)
compressed_32_test_data = compressed_model_32.predict(test_data)
compressed_32_val_data = compressed_model_32.predict(val_data)


# In[41]:


initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Define input shape
inputs = Input(shape=(32,))

# Define hidden layers with tanh activation
x = Dense(512, activation='tanh',kernel_initializer=initializer)(inputs)
x = Dense(128, activation='tanh',kernel_initializer=initializer)(x)
x = Dense(64, activation='tanh',kernel_initializer=initializer)(x)
# Define output layer with softmax activation
outputs = Dense(5, activation='softmax')(x)

# Define model
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
callback = EarlyStopping(monitor="loss",min_delta=0.0001, verbose = 0, restore_best_weights=True, patience = 2)
# Train the model
model_fit = model.fit(compressed_32_train_data, train_label, validation_data=(compressed_32_val_data, val_label), epochs=10000, batch_size=32,callbacks=callback,shuffle = True)



# In[42]:


# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(compressed_32_val_data, val_label)
print('Validation accuracy:', val_acc)
test_loss, test_acc = model.evaluate(compressed_32_test_data, test_label)
print('Test accuracy:', test_acc)


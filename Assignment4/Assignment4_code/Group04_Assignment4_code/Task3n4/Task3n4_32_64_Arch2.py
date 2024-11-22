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


# # Architecture 2: 512-128-64

# # Autoencoder 32 Hidden layer = 1

# In[28]:


initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
# Define input shape
inputs = Input(shape=(784,))

# Define encoding layer
autoencoder_32_train = Dense(32, activation='tanh',kernel_initializer=initializer)(inputs)

# Define decoding layer
decoded = Dense(784, activation='tanh',kernel_initializer=initializer)(autoencoder_32_train)

# Define autoencoder model
autoencoder_32 = Model(inputs=inputs, outputs=decoded)

# Compile the model
autoencoder_32.compile(optimizer='adam', loss='mse')
callback = EarlyStopping(monitor="val_loss",min_delta=0.0001, verbose = 0, restore_best_weights=True, patience = 5)
# Train the model
autoencoder_32.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=100000, batch_size=32,shuffle = True, callbacks = callback)

# Evaluate the model on the test set
test_loss1 = autoencoder_32.evaluate(test_data, test_data)
print('Average reconstruction error:', test_loss1)


# In[29]:


# Create a new model with the input as the original input and output as the output of the compressed layer
compressed_model_32 = Model(inputs=inputs, outputs=autoencoder_32_train)


# In[30]:


# Use the compressed model to extract the compressed representation of the training data
compressed_32_train_data = compressed_model_32.predict(train_data)
compressed_32_test_data = compressed_model_32.predict(test_data)
compressed_32_val_data = compressed_model_32.predict(val_data)


# In[31]:


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

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(compressed_32_val_data, val_label)
print('Validation accuracy:', val_acc)


# In[32]:


test_loss, test_acc = model.evaluate(compressed_32_test_data, test_label)
print('Test accuracy:', test_acc)


# In[33]:


predictions = model.predict(compressed_32_test_data, verbose=1)
p_class = np.argmax(predictions, axis=1)
# cm = tf.math.confusion_matrix(test_label, p_class)
cm = confusion_matrix(test_label, p_class)
cm_display = ConfusionMatrixDisplay(cm)
cm_display.plot()
plt.show()


# # Autoencoder 32 Hidden layer = 3

# In[34]:


initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
# Define input shape
inputs = Input(shape=(784,))

# Define encoding layers
encoded = Dense(400, activation='tanh',kernel_initializer=initializer)(inputs)
encoded_32_3 = Dense(32, activation='tanh',kernel_initializer=initializer)(encoded)

# Define decoding layers 
decoded = Dense(400, activation='tanh',kernel_initializer=initializer)(encoded_32_3)

# Define output layer 
decoded = Dense(784, activation='tanh',kernel_initializer=initializer)(decoded)

# Define autoencoder model
autoencoder_32_3 = Model(inputs=inputs, outputs=decoded)

# Compile the model
autoencoder_32_3.compile(optimizer='adam', loss='mse')
callback = EarlyStopping(monitor="val_loss",min_delta=0.0001, verbose = 0, restore_best_weights=True, patience = 5)
# Train the model
autoencoder_32_3.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=10, batch_size=32,callbacks = callback, shuffle = True)



# Evaluate the model on the test set
test_loss2 = autoencoder_32_3.evaluate(test_data, test_data)
print('Average reconstruction error:', test_loss2)


# In[35]:


# Create a new model with the input as the original input and output as the output of the compressed layer
compressed_model_32_3 = Model(inputs=inputs, outputs=encoded_32_3)


# In[36]:


# Use the compressed model to extract the compressed representation of the training data
compressed_32_3_train_data = compressed_model_32_3.predict(train_data)
compressed_32_3_test_data = compressed_model_32_3.predict(test_data)
compressed_32_3_val_data = compressed_model_32_3.predict(val_data)


# In[37]:


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
model_fit = model.fit(compressed_32_3_train_data, train_label, validation_data=(compressed_32_3_val_data, val_label), epochs=10000, batch_size=32,callbacks=callback,shuffle = True)

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(compressed_32_3_val_data, val_label)
print('Validation accuracy:', val_acc)


# In[38]:


test_loss, test_acc = model.evaluate(compressed_32_3_test_data, test_label)
print('Test accuracy:', test_acc)


# In[39]:


predictions = model.predict(compressed_32_3_test_data, verbose=1)
p_class = np.argmax(predictions, axis=1)
# cm = tf.math.confusion_matrix(test_label, p_class)
cm = confusion_matrix(test_label, p_class)
cm_display = ConfusionMatrixDisplay(cm)
cm_display.plot()
plt.show()


# # Autoencoder 64 Hidden layer = 1

# In[20]:


initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
# Define input shape
inputs = Input(shape=(784,))

# Define encoding layers
encoded_64 = Dense(64, activation='tanh',kernel_initializer=initializer)(inputs)

# Define decoding layers 

# Define output layer 
decoded = Dense(784, activation='tanh',kernel_initializer=initializer)(encoded_64)

# Define autoencoder model
autoencoder_64_1 = Model(inputs=inputs, outputs=decoded)

# Compile the model
autoencoder_64_1.compile(optimizer='adam', loss='mse')
callback = EarlyStopping(monitor="val_loss",min_delta=0.0001, verbose = 0, restore_best_weights=True, patience = 5)
# Train the model
autoencoder_64_1.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=10, batch_size=32,callbacks = callback, shuffle = True)



# Evaluate the model on the test set
test_loss2 = autoencoder_64_1.evaluate(test_data, test_data)
print('Average reconstruction error:', test_loss2)


# In[21]:


# Create a new model with the input as the original input and output as the output of the compressed layer
compressed_model_64_1 = Model(inputs=inputs, outputs=encoded_64)
# Use the compressed model to extract the compressed representation of the training data
compressed__64_1_train_data = compressed_model_64_1.predict(train_data)
compressed_64_1_test_data = compressed_model_64_1.predict(test_data)
compressed_64_1_val_data = compressed_model_64_1.predict(val_data)


# In[22]:


initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Define input shape
inputs = Input(shape=(64,))

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
model_fit = model.fit(compressed__64_1_train_data, train_label, validation_data=(compressed_64_1_val_data, val_label), epochs=10000, batch_size=32,callbacks=callback,shuffle = True)

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(compressed_64_1_val_data, val_label)
print('Validation accuracy:', val_acc)


# In[23]:


test_loss, test_acc = model.evaluate(compressed_64_1_test_data, test_label)
print('Test accuracy:', test_acc)


# # Autoencoder 64 Hidden layer = 3

# In[40]:


initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
# Define input shape
inputs = Input(shape=(784,))

# Define encoding layers
encoded = Dense(400, activation='tanh',kernel_initializer=initializer)(inputs)
encoded_64_3 = Dense(64, activation='tanh',kernel_initializer=initializer)(encoded)

# Define decoding layers 
decoded = Dense(400, activation='tanh',kernel_initializer=initializer)(encoded_64_3)

# Define output layer 
decoded = Dense(784, activation='tanh',kernel_initializer=initializer)(decoded)

# Define autoencoder model
autoencoder_64_3 = Model(inputs=inputs, outputs=decoded)

# Compile the model
autoencoder_64_3.compile(optimizer='adam', loss='mse')
callback = EarlyStopping(monitor="val_loss",min_delta=0.0001, verbose = 0, restore_best_weights=True, patience = 5)
# Train the model
autoencoder_64_3.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=10, batch_size=32,callbacks = callback, shuffle = True)



# Evaluate the model on the test set
test_loss2 = autoencoder_64_3.evaluate(test_data, test_data)
print('Average reconstruction error:', test_loss2)


# In[41]:


# Create a new model with the input as the original input and output as the output of the compressed layer
compressed_model_64_3 = Model(inputs=inputs, outputs=encoded_64_3)
# Use the compressed model to extract the compressed representation of the training data
compressed_64_3_train_data = compressed_model_64_3.predict(train_data)
compressed_64_3_test_data = compressed_model_64_3.predict(test_data)
compressed_64_3_val_data = compressed_model_64_3.predict(val_data)


# In[42]:


initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Define input shape
inputs = Input(shape=(64,))

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
model_fit = model.fit(compressed_64_3_train_data, train_label, validation_data=(compressed_64_3_val_data, val_label), epochs=10000, batch_size=32,callbacks=callback,shuffle = True)

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(compressed_64_3_val_data, val_label)
print('Validation accuracy:', val_acc)


# In[43]:


test_loss, test_acc = model.evaluate(compressed_64_3_test_data, test_label)
print('Test accuracy:', test_acc)


# In[44]:


predictions = model.predict(compressed_64_3_test_data, verbose=1)
p_class = np.argmax(predictions, axis=1)
# cm = tf.math.confusion_matrix(test_label, p_class)
cm = confusion_matrix(test_label, p_class)
cm_display = ConfusionMatrixDisplay(cm)
cm_display.plot()
plt.show()


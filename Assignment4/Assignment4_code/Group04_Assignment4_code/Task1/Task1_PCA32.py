#!/usr/bin/env python
# coding: utf-8

# # import important libraries
# 

# In[15]:


import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA


# In[16]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import initializers


# # creating labels for the dataset with digits 1,3,6,7,9

# In[17]:


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


# In[18]:


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


# In[19]:


path = "C:/Users/shilp/OneDrive/Documents/CS671/Group_4/"
train_data, test_data, val_data, train_out, test_out, val_out = read_data(path)


# In[20]:


train_label = convert(train_out)
val_label = convert(val_out)
test_label = convert(test_out)


# In[21]:


print(test_data.shape)


# In[22]:


# Flatten each image to a 784-dimensional vector
train_data = train_data.reshape((-1, 784))
val_data = val_data.reshape((-1, 784))
test_data = test_data.reshape((-1, 784))


# # perform PCA_ dim = 32

# In[23]:


# Perform PCA dimensionality reduction to 32 dimensions
pca = PCA(n_components=32)
train_data_pca = pca.fit_transform(train_data)
val_data_pca = pca.transform(val_data)
test_data_pca = pca.transform(test_data)


# # Architecture 1: 64-32-64

# In[24]:


initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)


# In[25]:


from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Define input shape
inputs = Input(shape=(32,))

# Define hidden layers with tanh activation
x = Dense(64, activation='tanh',kernel_initializer=initializer)(inputs)
x = Dense(32, activation='tanh',kernel_initializer=initializer)(x)
x = Dense(64, activation='tanh',kernel_initializer=initializer)(x)
# Define output layer with softmax activation
outputs = Dense(5, activation='softmax')(x)

# Define model
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
callback = EarlyStopping(monitor="loss",min_delta=0.0001, verbose = 0, restore_best_weights=True, patience = 2)
# Train the model
model_fit = model.fit(train_data_pca, train_label, validation_data=(val_data_pca, val_label), epochs=10000, batch_size=32,callbacks=callback,shuffle = True)



# In[26]:


# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(val_data_pca, val_label)
print('Validation accuracy:', val_acc)


# In[27]:


plt.plot(model_fit.history['loss'])
plt.title("Average Training Error Vs Epoch")
plt.xlabel("epochs")
plt.ylabel("Average error")
plt.show()


# In[28]:


loss, mse = model.evaluate(test_data_pca, test_label)


# In[29]:


predictions = model.predict(test_data_pca, verbose=1)
p_class = np.argmax(predictions, axis=1)
# cm = tf.math.confusion_matrix(test_label, p_class)
cm = confusion_matrix(test_label, p_class)
cm_display = ConfusionMatrixDisplay(cm)
cm_display.plot()
plt.show()


# # Architecture 2: 512-128-64

# In[30]:


initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)


# In[31]:


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
model_fit = model.fit(train_data_pca, train_label, validation_data=(val_data_pca, val_label), epochs=10000, batch_size=32,callbacks=callback,shuffle = True)

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(val_data_pca, val_label)
print('Validation accuracy:', val_acc)


# In[32]:


plt.plot(model_fit.history['loss'])
plt.title("Average Training Error Vs Epoch")
plt.xlabel("epochs")
plt.ylabel("Average error")
plt.show()


# In[36]:


loss, mse = model.evaluate(test_data_pca, test_label)


# In[35]:


predictions = model.predict(test_data_pca, verbose=1)
p_class = np.argmax(predictions, axis=1)
cm = confusion_matrix(test_label, p_class)
cm_display = ConfusionMatrixDisplay(cm)
cm_display.plot()
plt.show()


# # Architecture 3: 512-512-256-256-256

# In[35]:


initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)


# In[39]:


from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Define input shape
inputs = Input(shape=(32,))

# Define hidden layers with tanh activation
x = Dense(512, activation='tanh',kernel_initializer=initializer)(inputs)
x = Dense(512, activation='tanh',kernel_initializer=initializer)(x)
x = Dense(256, activation='tanh',kernel_initializer=initializer)(x)
x = Dense(256, activation='tanh',kernel_initializer=initializer)(x)
x = Dense(256, activation='tanh',kernel_initializer=initializer)(x)
# Define output layer with softmax activation
outputs = Dense(5, activation='softmax')(x)

# Define model
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
callback = EarlyStopping(monitor="loss",min_delta=0.0001, verbose = 0, restore_best_weights=True, patience = 2)
# Train the model
model_fit = model.fit(train_data_pca, train_label, validation_data=(val_data_pca, val_label), epochs=10000, batch_size=32,callbacks=callback,shuffle = True)

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(val_data_pca, val_label)
print('Validation accuracy:', val_acc)


# In[40]:


plt.plot(model_fit.history['loss'])
plt.title("Average Training Error Vs Epoch")
plt.xlabel("epochs")
plt.ylabel("Average error")
plt.show()


# In[41]:


loss, mse = model.evaluate(test_data_pca, test_label)


# In[42]:


predictions = model.predict(test_data_pca, verbose=1)
p_class = np.argmax(predictions, axis=1)
cm = confusion_matrix(test_label, p_class)
cm_display = ConfusionMatrixDisplay(cm)
cm_display.plot()
plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[69]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import initializers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense


# In[70]:


import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA


# # Read images and Resize image to 224x224

# In[71]:


def read_data(path):
    
    train_path = path+"/train"
    test_path = path+"/test"
    validation_path = path+"/val"
    # Define the new image size
    new_size = (224, 224)
    tr_data, test_data, val_data = [], [], []
    tr_out, test_out, val_out = [], [], []

    for i in os.listdir(train_path):

        if i != ".DS_Store":
            for j in os.listdir(train_path+"/"+i):
                tr_out.append(i)
                img = cv2.imread(train_path+"/"+i+"/"+j)
                # Resize the image
                resized_img = cv2.resize(img, new_size)
                tr_data.append(resized_img)
            for j in os.listdir(test_path+"/"+i):
                test_out.append(i)
                img = cv2.imread(test_path+"/"+i+"/"+j)
                # Resize the image
                resized_img = cv2.resize(img, new_size)
                test_data.append(resized_img)
            for j in os.listdir(validation_path+"/"+i):
                val_out.append(i)
                img = cv2.imread(validation_path+"/"+i+"/"+j)
                # Resize the image
                resized_img = cv2.resize(img, new_size)
                val_data.append(resized_img)
                
    tr_data, test_data, val_data = np.array(tr_data), np.array(test_data), np.array(val_data)
    tr_out, test_out, val_out = np.array(list(map(int, tr_out))), np.array(list(map(int, test_out))), np.array(list(map(int, val_out)))

    return tr_data, test_data, val_data, tr_out, test_out, val_out


# In[72]:


path = "C:/Users/shilp/OneDrive/Documents/CS671/A5_data/"
train_data, test_data, val_data, train_out, test_out, val_out = read_data(path)


# In[73]:


# # Normalize the pixel values to [0, 1]
# train_data = train_data / 255.0
# val_data = val_data / 255.0
# test_data = test_data / 255.0


# In[74]:


def convert(data):
  out = []
  for i in data:
    if i == 0:
      out.append(0)
    if i == 1:
      out.append(1)
    if i == 2:
      out.append(2)
    if i == 3:
      out.append(3)
    if i == 4:
      out.append(4)

  return np.array(out)


# In[75]:


train_label = convert(train_out)
val_label = convert(val_out)
test_label = convert(test_out)


# In[76]:


# Input layer
inputs = Input(shape=(224, 224, 3))

# First convolutional layer
conv1 = Conv2D(8, (11, 11), strides=4, padding='valid', activation='relu')(inputs)
pool1 = MaxPooling2D((3, 3), strides=2)(conv1)

# Second convolutional layer
conv2 = Conv2D(16, (5, 5), strides=1, padding='valid', activation='relu')(pool1)
pool2 = MaxPooling2D((3, 3), strides=2)(conv2)

# Flatten the output from the convolutional layers
flatten = Flatten()(pool2)

# Fully connected layers
fc1 = Dense(128, activation='relu')(flatten)
output = Dense(5, activation='softmax')(fc1)

# Define the model
model = Model(inputs=inputs, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Save the model weights
# model.save_weights("C:/Users/shilp/OneDrive/Documents/CS671/architecture_1_weights.h5")


# In[77]:


callback = EarlyStopping(monitor="val_loss",min_delta=0.0001, verbose = 0, restore_best_weights=True, patience = 2)
# Train the model
model.fit(train_data, train_label, validation_data=(val_data, val_label), epochs=10000, batch_size=32,callbacks=callback,shuffle = True)
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_data, test_label)
print('Test accuracy:', test_acc)


# In[78]:


# Evaluate the model on the val set
val_loss, val_acc = model.evaluate(val_data, val_label)
print('Val accuracy:', val_acc)


# In[79]:


# Evaluate the model on the train set
train_loss, train_acc = model.evaluate(train_data, train_label)
print('train accuracy:', train_acc)


# In[80]:


model.summary()


# # Architecture 2

# In[92]:


# Input layer
inputs = Input(shape=(224, 224, 3))

# First convolutional layer
conv1 = Conv2D(8, (11, 11), strides=4, padding='valid', activation='relu')(inputs)
pool1 = MaxPooling2D((3, 3), strides=2)(conv1)

# Second convolutional layer
conv2 = Conv2D(16, (5, 5), strides=1, padding='valid', activation='relu')(pool1)
pool2 = MaxPooling2D((3, 3), strides=2)(conv2)

# Third convolutional layer
conv3 = Conv2D(32, (3,3), strides=1, padding='valid', activation='relu')(pool2)
pool3 = MaxPooling2D((3, 3), strides=2)(conv3)

# Flatten the output from the convolutional layers
flatten = Flatten()(pool3)

# Fully connected layers
fc1 = Dense(128, activation='relu')(flatten)
output = Dense(5, activation='softmax')(fc1)

# Define the model
model_2 = Model(inputs=inputs, outputs=output)

# Compile the model
model_2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



# In[82]:


callback = EarlyStopping(monitor="val_loss",min_delta=0.0001, verbose = 0, restore_best_weights=True, patience = 2)
# Train the model
model_2.fit(train_data, train_label, validation_data=(val_data, val_label), epochs=10000, batch_size=32,callbacks=callback,shuffle = True)
# Evaluate the model on the test set
test_loss, test_acc = model_2.evaluate(test_data, test_label)
print('Test accuracy:', test_acc)
# Evaluate the model on the val set
val_loss, val_acc = model_2.evaluate(val_data, val_label)
print('Val accuracy:', val_acc)
# Evaluate the model on the train set
train_loss, train_acc = model_2.evaluate(train_data, train_label)
print('train accuracy:', train_acc)
model_2.save_weights("C:/Users/shilp/OneDrive/Documents/CS671/architecture_2_weights_w.h5")
model.save('C:/Users/shilp/OneDrive/Documents/CS671/architecture_2_w_model.h5')


# In[83]:


model_2.summary()


# In[93]:


# Create a new model that outputs the feature maps from the first , second and third convolutional layers
model_new = Model(inputs=inputs, outputs=[conv1, conv2, conv3])


# In[94]:


# Select a random image from the training set
x = train_data[0]

# Predict the feature maps for the selected image
feature_maps = model_new.predict(x.reshape(1, 224, 224, 3))


# In[95]:


# Plot the feature maps from the first convolutional layer
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(10,5))
for i in range(8):
    row = i // 4
    col = i % 4
    axs[row, col].imshow(feature_maps[0][0, :, :, i], cmap='cubehelix')
plt.suptitle('Feature maps from first convolutional layer')
plt.show()

# Plot the feature maps from the second convolutional layer
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(10,5))
for i in range(8):
    row = i // 4
    col = i % 4
    axs[row, col].imshow(feature_maps[1][0, :, :, i], cmap='cubehelix')
plt.suptitle('Feature maps from second convolutional layer')
plt.show()
# Plot the feature maps from the third convolutional layer
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(10,5))
for i in range(8):
    row = i // 4
    col = i % 4
    axs[row, col].imshow(feature_maps[2][0, :, :, i], cmap='cubehelix')
plt.suptitle('Feature maps from third convolutional layer')
plt.show()


# # Architecture 3

# In[84]:


# Input layer
inputs = Input(shape=(224, 224, 3))

# First convolutional layer
conv1 = Conv2D(8, (11, 11), strides=4, padding='valid', activation='relu')(inputs)
pool1 = MaxPooling2D((3, 3), strides=2)(conv1)

# Second convolutional layer
conv2 = Conv2D(16, (5, 5), strides=1, padding='valid', activation='relu')(pool1)
pool2 = MaxPooling2D((3, 3), strides=2)(conv2)

# Third convolutional layer
conv3 = Conv2D(32, (3,3), strides=1, padding='valid', activation='relu')(pool2)
# pool3 = MaxPooling2D((3, 3), strides=2)(conv3)

# Fourth convolutional layer
conv4 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='valid')(conv3)
maxpool4 = MaxPooling2D((3, 3), strides=2)(conv4)

# Flatten the output from the convolutional layers
flatten = Flatten()(maxpool4)

# Fully connected layers
fc1 = Dense(128, activation='relu')(flatten)
output = Dense(5, activation='softmax')(fc1)

# Define the model
model_3 = Model(inputs=inputs, outputs=output)

# Compile the model
model_3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



# In[85]:


callback = EarlyStopping(monitor="val_loss",min_delta=0.0001, verbose = 0, restore_best_weights=True, patience = 2)
# Train the model
model_3.fit(train_data, train_label, validation_data=(val_data, val_label), epochs=10000, batch_size=32,callbacks=callback,shuffle = True)
# Evaluate the model on the test set
test_loss, test_acc = model_3.evaluate(test_data, test_label)
print('Test accuracy:', test_acc)
# Evaluate the model on the val set
val_loss, val_acc = model_3.evaluate(val_data, val_label)
print('Val accuracy:', val_acc)
# Evaluate the model on the train set
train_loss, train_acc = model_3.evaluate(train_data, train_label)
print('train accuracy:', train_acc)


# In[86]:


predictions = model_3.predict(test_data, verbose=1)
p_class = np.argmax(predictions, axis=1)
# cm = tf.math.confusion_matrix(test_label, p_class)
cm = confusion_matrix(test_label, p_class)
cm_display = ConfusionMatrixDisplay(cm)
cm_display.plot()
plt.show()


# In[87]:


model_3.summary()


# In[88]:


plt.imshow(train_data[0])


# In[89]:


from keras.models import Model
import matplotlib.pyplot as plt

# Select a random image from the training set
x = train_data[0]

# Create a new model that outputs the feature maps from the first and third convolutional layers
model_3_layers = Model(inputs=model_3.input, outputs=[model_3.layers[1].output, model_3.layers[3].output])

# Predict the feature maps for the selected image
feature_maps = model_3_layers.predict(x.reshape(1, 224, 224, 3))

# Plot the feature maps from the first convolutional layer
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(10,5))
for i in range(8):
    row = i // 4
    col = i % 4
    axs[row, col].imshow(feature_maps[0][0, :, :, i], cmap='cubehelix')
plt.suptitle('Feature maps from first convolutional layer')
plt.show()

# Plot the feature maps from the third convolutional layer
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(10,5))
for i in range(8):
    row = i // 4
    col = i % 4
    axs[row, col].imshow(feature_maps[1][0, :, :, i], cmap='cubehelix')
plt.suptitle('Feature maps from second convolutional layer')
plt.show()


# In[91]:


# plt.imshow(train_data[1])
print(train_data[1].shape)


# In[ ]:





# In[ ]:





# In[ ]:





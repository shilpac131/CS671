#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow.keras as keras
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import pickle as p
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Input, Masking


# # prep data

# In[2]:


file_path = "C:/Users/shilp/OneDrive/Documents/CS671/A6_data/q2"
classes = ['hI', 'ne', 'ni', 'nii', 'pa']
data_split = ['Test', 'Train']


# In[3]:


test = []
train = []
test_label = []
train_label = []

dir_list = os.listdir(file_path)
print(dir_list)
for j in data_split:
    for i in classes:
        for filename in glob.glob(os.path.join(file_path + '/' + i + '/' + j, '*.mfcc')):
            with open(filename, 'r') as f:
                if (j == "Train"):
                    # print(filename)
                    f = open(filename)
                    parsed = f.readlines()
                    parsed = [np.array([float(val) for val in line.split(" ") if val!="\n"]) for line in parsed]
                    train.append(np.array(parsed))
                    train_label.append(str(i))
                if (j == 'Test'):
                    # print(filename)
                    f = open(filename)
                    parsed = f.readlines()
                    parsed = [np.array([float(val) for val in line.split(" ") if val!="\n"]) for line in parsed]
                    test.append(np.array(parsed))
                    test_label.append(str(i))
print("Training Set has {} samples each with dimension {} but variable length".format(len(train),train[0][0].shape))
print("Test Set has {} samples each with dimension {} but variable length".format(len(test),test[0][0].shape))
print("type = ",type(train))

# padding input
max_len = max([len(val) for val in train])
print("Max_seq length: ",max_len)
def padd_it(data):
    for i in range(len(data)):
        if len(data[i]) <= max_len:
            seq = np.repeat(np.array([-1]*39).reshape(-1,39),int(max_len-len(data[i])),0)
            data[i] = np.append(data[i],seq,axis=0)
    return data
# data preprocessing steps to prepare the input data for training and testing a model
print("Before Padding shape of first element in train is {}".format((train[2]).shape))
test_pad = np.asarray(padd_it(test))
train_pad = np.asarray(padd_it(train))
print("After Padding shape of first element in train is {}".format((train_pad[0]).shape))


# In[4]:


print(test_label[0])


# In[5]:


train_pad.shape[1:]


# In[6]:


print("Training set shape ",train_pad.shape)
print("Test set shape: ",test_pad.shape)
def one_hot_code(data):
    new_data = []
    for i in range(len(data)):
        index = classes.index(data[i])
        l = [0]*5
        l[index]=1
        l = np.array(l)
        new_data.append(l)
    return np.array(new_data)
train_label = one_hot_code(train_label)
test_label = one_hot_code(test_label)
print("train target shape, ",test_label.shape)


# In[7]:


train_pad.shape[1:]


# ## model 1

# In[8]:


model=Sequential()
model.add(Masking(mask_value=-1,input_shape=train_pad.shape[1:]))  # to mask values other then -1 (padded value that shall be ignored)
model.add(SimpleRNN(256, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))
model.summary()


# In[9]:


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor="loss", min_delta=0.0001,patience=3)

history = model.fit(train_pad, train_label, batch_size=64,epochs = 6000,verbose=1, validation_data=(test_pad, test_label),callbacks=callback)


# In[10]:


plt.plot(history.history['loss'])
plt.title("Average Training Error Vs Epoch")
plt.xlabel("epochs")
plt.ylabel("Average error")
plt.show()


# In[11]:


train_loss, train_acc = model.evaluate(train_pad,train_label)
print('train accuracy:', train_acc)
test_loss, test_acc = model.evaluate(test_pad,test_label)
print('Test accuracy:', test_acc)


# In[12]:


predict = model.predict(test_pad)
pred_test = np.argmax(predict, axis=1)
true_test = np.argmax(test_label, axis=1)

cm_test = confusion_matrix(true_test, pred_test)
# print(cm_test)
cm_display = ConfusionMatrixDisplay(cm_test, display_labels=classes)
cm_display.plot()
plt.show()


# ## model 2

# In[13]:


model=Sequential()
model.add(Masking(mask_value=-1,input_shape=train_pad.shape[1:]))  # to mask values other then -1 (padded value that shall be ignored)
model.add(SimpleRNN(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(SimpleRNN(256, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))
model.summary()


# In[14]:


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor="loss", min_delta=0.0001,patience=3)

history = model.fit(train_pad, train_label, batch_size=64,epochs = 6000,verbose=1, validation_data=(test_pad, test_label),callbacks=callback)


# In[15]:


plt.plot(history.history['loss'])
plt.title("Average Training Error Vs Epoch")
plt.xlabel("epochs")
plt.ylabel("Average error")
plt.show()


# In[16]:


train_loss, train_acc = model.evaluate(train_pad,train_label)
print('train accuracy:', train_acc)
test_loss, test_acc = model.evaluate(test_pad,test_label)
print('Test accuracy:', test_acc)


# # model 3

# In[24]:


model=Sequential()
model.add(Masking(mask_value=-1,input_shape=train_pad.shape[1:]))  # to mask values other then -1 (padded value that shall be ignored)
model.add(SimpleRNN(128, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(5, activation='softmax'))
model.summary()


# In[25]:


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor="loss", min_delta=0.0001,patience=2)

history = model.fit(train_pad, train_label, batch_size=64,epochs = 6000,verbose=1, validation_data=(test_pad, test_label),callbacks=callback)


# In[26]:


plt.plot(history.history['loss'])
plt.title("Average Training Error Vs Epoch")
plt.xlabel("epochs")
plt.ylabel("Average error")
plt.show()


# In[27]:


train_loss, train_acc = model.evaluate(train_pad,train_label)
print('train accuracy:', train_acc)
test_loss, test_acc = model.evaluate(test_pad,test_label)
print('Test accuracy:', test_acc)


# # model 4

# In[29]:


model=Sequential()
model.add(Masking(mask_value=-1,input_shape=train_pad.shape[1:]))  # to mask values other then -1 (padded value that shall be ignored)
model.add(SimpleRNN(64, return_sequences=True))
model.add(SimpleRNN(64, return_sequences=False))
model.add(Dense(5, activation='softmax'))
model.summary()


# In[30]:


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor="loss", min_delta=0.0001,patience=2)

history = model.fit(train_pad, train_label, batch_size=64,epochs = 6000,verbose=1, validation_data=(test_pad, test_label),callbacks=callback)


# In[31]:


plt.plot(history.history['loss'])
plt.title("Average Training Error Vs Epoch")
plt.xlabel("epochs")
plt.ylabel("Average error")
plt.show()


# In[32]:


train_loss, train_acc = model.evaluate(train_pad,train_label)
print('train accuracy:', train_acc)
test_loss, test_acc = model.evaluate(test_pad,test_label)
print('Test accuracy:', test_acc)


# # model 5
# 

# In[33]:


model=Sequential()
model.add(Masking(mask_value=-1,input_shape=train_pad.shape[1:]))  # to mask values other then -1 (padded value that shall be ignored)
model.add(SimpleRNN(32, return_sequences=True))
model.add(Dropout(0.2))
model.add(SimpleRNN(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))
model.summary()


# In[34]:


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor="loss", min_delta=0.0001,patience=2)

history = model.fit(train_pad, train_label, batch_size=64,epochs = 6000,verbose=1, validation_data=(test_pad, test_label),callbacks=callback)


# In[35]:


plt.plot(history.history['loss'])
plt.title("Average Training Error Vs Epoch")
plt.xlabel("epochs")
plt.ylabel("Average error")
plt.show()


# In[53]:


train_loss, train_acc = model.evaluate(train_pad,train_label)
print('train accuracy:', train_acc)
test_loss, test_acc = model.evaluate(test_pad,test_label)
print('Test accuracy:', test_acc)


# In[54]:


predict = model.predict(test_pad)
pred_test = np.argmax(predict, axis=1)
true_test = np.argmax(test_label, axis=1)

cm_test = confusion_matrix(true_test, pred_test)
# print(cm_test)
cm_display = ConfusionMatrixDisplay(cm_test, display_labels=classes)
cm_display.plot()
plt.show()



# In[ ]:





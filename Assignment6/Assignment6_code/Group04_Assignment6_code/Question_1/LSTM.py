#!/usr/bin/env python
# coding: utf-8

# ## import all libraries

# In[1]:


import os
import glob
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
from keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Input, Masking
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# # prep data

# In[3]:


file_path = "C:/Users/Deepak Angra/Downloads/a6_data"
classes = ['a', 'ai', 'bA', 'chA', 'dA']
data_split = ['dev', 'train']


# In[4]:


train = []
test = []

dir_list = os.listdir(file_path)
print(dir_list)


# In[5]:


train_label = []
test_label = []
for j in data_split:
    for k in classes:
        for filename in glob.glob(os.path.join(file_path + '/' + k + '/' + j, '*.txt')):
            with open(filename, 'r') as f:
                data = f.read().split()
                length = int(data[0])
                if (j == "train"):
                    points = []
                    for i in range(1, length*2, 2):
                        x = float(data[i])
                        y = float(data[i+1])
                        points.append((x, y))
                    train.append(np.array(points))
                    train_label.append(str(k))

                elif (j == 'dev'):
                        points = []
                        length_train = []
                        for i in range(1, length*2, 2):
                            x = float(data[i])
                            y = float(data[i+1])
                            points.append((x, y))
                        test.append(np.array(points))
                        test_label.append(str(k))


# In[6]:


print(len(train[54]))
print(test_label[54])


# In[7]:


# train[0]


# In[8]:


print("Training Set has {} samples each with dimension {} but variable length".format(len(train),train[0].shape))
print("Test Set has {} samples each with dimension {} but variable length".format(len(test),test[0].shape))
print("type = ",type(test))


# In[9]:


# padding input
argmax_len = np.argmax([len(val) for val in train])
max_len = max([len(val) for val in train])
print("Max seq index: ",max_len)
print("Max seq : ", argmax_len)


# In[10]:


# padding input
argmax_len_test = np.argmax([len(val) for val in test])
max_len_test = max([len(val) for val in test])
print("Max seq index: ",max_len_test)
print("Max seq : ", argmax_len_test)


# In[11]:


len(train[321])


# # pad (2,2)

# In[12]:


data_pad  = []
train_pad = []
def padd_it(data, max_len):
    for i in range(len(data)):
        if len(data[i]) <= max_len:
            pad_arr = []
            padded_arr = np.pad(data[i], ((0, max_len-len(data[i])), (0, 0)), mode='constant', constant_values=(2,2))
            data_pad.append(np.array(padded_arr))
    return data_pad
# data preprocessing steps to prepare the input data for training and testing a model
print("Before Padding shape of first element in train is {}".format((train[2]).shape))
train_pad = np.asarray(padd_it(train,max_len ))
print("After Padding shape of first element in train is {}".format((train_pad[2]).shape))


# In[13]:


data_pad  = []
test_pad = []
def padd_it(data, max_len):
    for i in range(len(data)):
        if len(data[i]) <= max_len:
            pad_arr = []
            padded_arr = np.pad(data[i], ((0, max_len-len(data[i])), (0, 0)), mode='constant', constant_values=(2,2))
            data_pad.append(np.array(padded_arr))
    return data_pad
test_pad = np.asarray(padd_it(test ,max_len))
# data preprocessing steps to prepare the input data for training and testing a model
print("Before Padding shape of first element in train is {}".format((test[2]).shape))
print("After Padding shape of first element in train is {}".format((test_pad[2]).shape))


# In[14]:


print(len(test_pad))
# print(len(train_pad))


# In[15]:


test_label[0]


# In[16]:


print("Training set shape ",train_pad.shape)
print("Test set shape: ",test_pad.shape)


# In[17]:


len(test_pad[0])


# # normalize

# In[18]:


# Compute the maximum value in each column after excluding the points with value 2
max_x = np.max(train_pad[train_pad[:,:,0] != 2][:,0])
max_y = np.max(train_pad[train_pad[:,:,1] != 2][:,1])
# Normalize the data in each column using the maximum value
train_pad[:, :, 0] = np.where(train_pad[:, :, 0] == 2, train_pad[:, :, 0], train_pad[:, :, 0]/max_x)
train_pad[:, :, 1] = np.where(train_pad[:, :, 1] == 2, train_pad[:, :, 1], train_pad[:, :, 1]/max_y)


# In[19]:


# Normalize the data in each column using the maximum value
test_pad[:, :, 0] = np.where(test_pad[:, :, 0] == 2, test_pad[:, :, 0], test_pad[:, :, 0]/max_x)
test_pad[:, :, 1] = np.where(test_pad[:, :, 1] == 2, test_pad[:, :, 1], test_pad[:, :, 1]/max_y)


# In[20]:


print(max_x)
print(max_y)


# In[21]:


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
print("train target shape, ",train_label.shape)


# In[22]:


test_label = one_hot_code(test_label)
print("test target shape, ",test_label.shape)


# In[23]:


train_pad.shape[1:]


# # model 1

# In[24]:


model1=Sequential()
model1.add(Masking(mask_value=2,input_shape=train_pad.shape[1:]))  # to mask values other then 2 (padded value that shall be ignored)
model1.add(LSTM(64, return_sequences=True))
model1.add(Dropout(0.2))
model1.add(LSTM(32, return_sequences=True))
model1.add(Dropout(0.2))
model1.add(LSTM(32, return_sequences=False))
model1.add(Dropout(0.2))
model1.add(Dense(5, activation='softmax'))
model1.summary()


# In[25]:


model1.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=11, restore_best_weights = True)

history = model1.fit(train_pad, train_label, batch_size=32,epochs = 6000,verbose=1, validation_data=(test_pad, test_label),callbacks=callback)


# In[26]:


plt.plot(history.history['val_loss'])
plt.title("Average Training Error Vs Epoch")
plt.xlabel("epochs")
plt.ylabel("Average error")
plt.show()


# In[27]:


plt.plot(history.history['loss'])
plt.title("Average Training Error Vs Epoch")
plt.xlabel("epochs")
plt.ylabel("Average error")
plt.show()


# In[28]:


model1.evaluate(train_pad,train_label)


# In[29]:


model1.evaluate(test_pad,test_label)


# In[30]:


predictions = model1.predict(test_pad, verbose=1)
p_class = np.argmax(predictions, axis=1)


# In[31]:


test_label_int = np.argmax(test_label, axis=1)
cm = confusion_matrix(test_label_int, p_class)
print(cm)


# In[32]:


cm_display = ConfusionMatrixDisplay(cm)
cm_display.plot()
plt.show()


# In[ ]:





# 
# # Model 2

# In[33]:


model2=Sequential()
model2.add(Masking(mask_value=2,input_shape=train_pad.shape[1:]))  # to mask values other then 2 (padded value that shall be ignored)
model2.add(LSTM(128, return_sequences=True))
model2.add(Dropout(0.2))
model2.add(LSTM(32, return_sequences=True))
model2.add(Dropout(0.2))
model2.add(LSTM(32, return_sequences=False))
model2.add(Dropout(0.2))
model2.add(Dense(5, activation='softmax'))
model2.summary()


# In[34]:


model2.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=5, restore_best_weights = True)

history = model2.fit(train_pad, train_label, batch_size=32,epochs = 6000,verbose=1, validation_data=(test_pad, test_label),callbacks=callback)


# In[35]:


plt.plot(history.history['loss'])
plt.title("Average Training Error Vs Epoch")
plt.xlabel("epochs")
plt.ylabel("Average error")
plt.show()


# In[36]:


plt.plot(history.history['val_loss'])
plt.title("Average Training Error Vs Epoch")
plt.xlabel("epochs")
plt.ylabel("Average error")
plt.show()


# In[37]:


model2.evaluate(train_pad,train_label)


# In[38]:


model2.evaluate(test_pad,test_label)


# In[39]:


predictions = model2.predict(test_pad, verbose=1)
p_class = np.argmax(predictions, axis=1)
test_label_int = np.argmax(test_label, axis=1)
cm = confusion_matrix(test_label_int, p_class)
#print(cm)
cm_display = ConfusionMatrixDisplay(cm, display_labels=classes)
cm_display.plot()
plt.title("Confusion Matrix")
plt.show()


# # model 3

# In[40]:


model3=Sequential()
model3.add(Masking(mask_value=2,input_shape=train_pad.shape[1:]))  # to mask values other then 2 (padded value that shall be ignored)
model3.add(LSTM(128, return_sequences=True))
model3.add(Dropout(0.2))
model3.add(LSTM(32, return_sequences=False))
model3.add(Dropout(0.2))
model3.add(Dense(5, activation='softmax'))
model3.summary()


# In[41]:


model3.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=11, restore_best_weights = True)

history = model3.fit(train_pad, train_label, batch_size=32,epochs = 6000,verbose=1, validation_data=(test_pad, test_label),callbacks=callback)


# In[42]:


plt.plot(history.history['loss'])
plt.title("Average Training Error Vs Epoch")
plt.xlabel("epochs")
plt.ylabel("Average error")
plt.show()


# In[43]:


plt.plot(history.history['val_loss'])
plt.title("Average Training Error Vs Epoch")
plt.xlabel("epochs")
plt.ylabel("Average error")
plt.show()


# In[44]:


model3.evaluate(train_pad,train_label)


# In[45]:


model3.evaluate(test_pad,test_label)


# In[46]:


predictions = model3.predict(test_pad, verbose=1)
p_class = np.argmax(predictions, axis=1)
test_label_int = np.argmax(test_label, axis=1)
cm = confusion_matrix(test_label_int, p_class)
#print(cm)
cm_display = ConfusionMatrixDisplay(cm)
cm_display.plot()
plt.show()


# In[ ]:





# # model4

# In[47]:


model4=Sequential()
model4.add(Masking(mask_value=2,input_shape=train_pad.shape[1:]))  # to mask values other then 2 (padded value that shall be ignored)
model4.add(LSTM(64, return_sequences=True))
model4.add(Dropout(0.2))
model4.add(LSTM(64, return_sequences=False))
model4.add(Dropout(0.2))
model4.add(Dense(8,activation='relu'))
model4.add(Dense(5, activation='softmax'))
model4.summary()


# In[48]:


model4.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=11, restore_best_weights = True)

history = model4.fit(train_pad, train_label, batch_size=32,epochs = 6000,verbose=1, validation_data=(test_pad, test_label),callbacks=callback)


# In[49]:


plt.plot(history.history['loss'])
plt.title("Average Training Error Vs Epoch")
plt.xlabel("epochs")
plt.ylabel("Average error")
plt.show()


# In[50]:


plt.plot(history.history['val_loss'])
plt.title("Average Training Error Vs Epoch")
plt.xlabel("epochs")
plt.ylabel("Average error")
plt.show()


# In[51]:


model4.evaluate(train_pad,train_label)


# In[52]:


model4.evaluate(test_pad,test_label)


# In[53]:


predictions = model4.predict(test_pad, verbose=1)
p_class = np.argmax(predictions, axis=1)
test_label_int = np.argmax(test_label, axis=1)
cm = confusion_matrix(test_label_int, p_class)
#print(cm)
cm_display = ConfusionMatrixDisplay(cm)
cm_display.plot()
plt.show()


# In[ ]:





# # model5

# In[54]:


model5=Sequential()
model5.add(Masking(mask_value=2,input_shape=train_pad.shape[1:]))  # to mask values other then 2 (padded value that shall be ignored)
model5.add(LSTM(32, return_sequences=True))
model5.add(Dropout(0.2))
model5.add(LSTM(32, return_sequences=True))
model5.add(Dropout(0.2))
model5.add(LSTM(32, return_sequences=False))
model5.add(Dropout(0.2))
model5.add(Dense(5, activation='softmax'))
model5.summary()


# In[55]:


model5.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=11, restore_best_weights = True)

history = model5.fit(train_pad, train_label, batch_size=32,epochs = 6000,verbose=1, validation_data=(test_pad, test_label),callbacks=callback)


# In[56]:


plt.plot(history.history['loss'])
plt.title("Average Training Error Vs Epoch")
plt.xlabel("epochs")
plt.ylabel("Average error")
plt.show()


# In[57]:


plt.plot(history.history['val_loss'])
plt.title("Average Training Error Vs Epoch")
plt.xlabel("epochs")
plt.ylabel("Average error")
plt.show()


# In[58]:


model5.evaluate(train_pad,train_label)


# In[59]:


model5.evaluate(test_pad,test_label)


# In[60]:


predictions = model5.predict(test_pad, verbose=1)
p_class = np.argmax(predictions, axis=1)
test_label_int = np.argmax(test_label, axis=1)
cm = confusion_matrix(test_label_int, p_class)
#print(cm)
cm_display = ConfusionMatrixDisplay(cm)
cm_display.plot()
plt.show()


# In[ ]:





# In[ ]:





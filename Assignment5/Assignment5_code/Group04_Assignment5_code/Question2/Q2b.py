#!/usr/bin/env python
# coding: utf-8

# # Visualizing Patches: VGG19

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import initializers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras import backend as K
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg16 import preprocess_input


# In[2]:


import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


# # Read images and Resize image to 224x224

# In[3]:


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


# In[4]:


path = "C:/Users/shilp/OneDrive/Documents/CS671/A5_data/"
train_data, test_data, val_data, train_out, test_out, val_out = read_data(path)


# In[5]:


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


# In[6]:


train_label = convert(train_out)
val_label = convert(val_out)
test_label = convert(test_out)


# In[7]:


# Load the VGG19 model
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the weights of the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add a new classification layer with 5 output nodes
x = base_model.output
x = Flatten()(x)
x = Dense(5, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[8]:


from tensorflow.keras.callbacks import EarlyStopping

# Define the early stopping criteria
callback = EarlyStopping(monitor="val_loss", min_delta=0.0001, verbose=0, restore_best_weights=True, patience=2)

# Train the model
model.fit(train_data, train_label, epochs=10000, batch_size=32, callbacks=[callback], validation_data=(val_data, val_label))

# Evaluate the model on the training set
train_loss, train_acc = model.evaluate(train_data, train_label)
print('Training accuracy:', train_acc)

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(val_data, val_label)
print('Validation accuracy:', val_acc)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_data, test_label)
print('Test accuracy:', test_acc)


# In[9]:


model.summary()


# ### Sample Test Image

# In[188]:


image = cv2.imread('C:/Users/shilp/OneDrive/Documents/CS671/A5_data/train/4/image_0081.jpg')
image = cv2.resize(image, (224, 224))
plt.imshow(image)

image.shape


# ### Testing

# In[189]:


test = np.expand_dims(image, axis=0)
test.shape


# ### Traceback

# In[190]:


def get_feature(layer_name, test):
  layer_outputs = [layer.output for layer in model.layers if layer.name == layer_name]
  activation_model = Model(inputs=model.input, outputs=layer_outputs)
  return activation_model.predict(test)


# ##### Plotting one example

# In[191]:


last_conv = get_feature('block5_conv4' , test)


# In[192]:


for i in range(25):
  plt.subplot(5, 5, i+1)
  plt.imshow(np.squeeze(last_conv[:, :, :, i]))
plt.show()


# ##### Finding Maximally activated neuron in final conv layer

# In[193]:


max_pos = np.argmax(np.squeeze(last_conv[:, :, :, 1]))
max_pos


# In[194]:


np.ndarray.flatten(np.squeeze(last_conv[:, :, :, i]))[max_pos], np.amax(np.squeeze(last_conv[:, :, :, 1]))


# ##### Getting features of each layer w/o gradients

# In[195]:


block5conv4 = get_feature('block5_conv4', test)


# ##### Back tracing last three conv layers (Note the strides and padding)

# In[196]:


def trace_patch(kernel_size, stride, max_pos):
    '''
    This function returns the positions of the pixels in the previous feature map that correspond to the output pixel at max_pos.
    kernel_size: size of the convolution kernel
    stride: stride of the convolution
    max_pos: (x,y) coordinates of the output pixel in the current feature map
    '''
    padding = (kernel_size - 1) // 2 # compute the padding size needed to keep the spatial dimensions unchanged
    input_positions = []
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = (max_pos[0] * stride) - padding + j
            y = (max_pos[1] * stride) - padding + i
            input_positions.append((x,y))
    
    return input_positions


# In[197]:


#max_pos = np.argmax(np.squeeze(last_conv[:, :, :, 1]))
max_value = np.max(np.squeeze(last_conv[:, :, :, 1]))
print(max_value)
max_pos = np.where(np.squeeze(last_conv[:, :, :, 1]) == max_value)
max_pos = (max_pos[0][0], max_pos[1][0])
max_pos


# In[198]:


patch1 = trace_patch(3,1,max_pos)


# In[199]:


patch1


# In[200]:


patch2 = []
for i in patch1:
  patch2.extend(trace_patch(3,1,i))

patch2 = list(set(patch2)) #unique positions only
print(patch2)


# In[201]:


patch3 = []
for i in patch2:
  patch3.extend(trace_patch(2,2,i))

patch3 = list(set(patch3)) #unique positions only
print(patch3)


# In[202]:


patch4 = []
for i in patch3:
  patch4.extend(trace_patch(3,1,i))

patch4 = list(set(patch4)) #unique positions only
# print(patch4)


# In[203]:


patch5 = []
for i in patch4:
  patch5.extend(trace_patch(3,1,i))

patch5 = list(set(patch5)) #unique positions only
# print(patch5)


# In[204]:


patch6 = []
for i in patch5:
  patch6.extend(trace_patch(3,1,i))

patch6 = list(set(patch6)) #unique positions only
# print(patch6)


# In[205]:


patch7 = []
for i in patch6:
  patch7.extend(trace_patch(3,1,i))

patch7 = list(set(patch7)) #unique positions only
# print(patch7)


# In[206]:


patch8 = []
for i in patch7:
  patch8.extend(trace_patch(2,2,i))

patch8 = list(set(patch8)) #unique positions only


# In[207]:


patch9 = []
for i in patch8:
  patch9.extend(trace_patch(3,1,i))

patch9 = list(set(patch9)) #unique positions only

patch10 = []
for i in patch9:
  patch10.extend(trace_patch(3,1,i))

patch10 = list(set(patch10)) #unique positions only


# In[208]:


patch11 = []
for i in patch10:
  patch11.extend(trace_patch(3,1,i))

patch11 = list(set(patch11)) #unique positions only


# In[209]:


patch12 = []
for i in patch11:
  patch12.extend(trace_patch(3,1,i))

patch12 = list(set(patch12)) #unique positions only

patch13 = []
for i in patch12:
  patch13.extend(trace_patch(2,2,i))

patch13 = list(set(patch13)) #unique positions only


# In[210]:


patch14 = []
for i in patch13:
  patch14.extend(trace_patch(3,1,i))

patch14 = list(set(patch14)) #unique positions only
# print(patch7)


# In[211]:


patch15 = []
for i in patch14:
  patch15.extend(trace_patch(3,1,i))

patch15 = list(set(patch15)) #unique positions only

patch16 = []
for i in patch15:
  patch16.extend(trace_patch(2,2,i))

patch16 = list(set(patch16)) #unique positions only


# In[212]:


patch17 = []
for i in patch16:
  patch17.extend(trace_patch(3,1,i))

patch17 = list(set(patch17)) #unique positions only

patch18 = []
for i in patch17:
  patch18.extend(trace_patch(3,1,i))

patch18 = list(set(patch18)) #unique positions only


# In[213]:


def find_extreme_pixels(pixel_points):
    # initialize variables to store extreme pixels
    bottom_left = pixel_points[0]
    bottom_right = pixel_points[0]
    top_left = pixel_points[0]
    top_right = pixel_points[0]

    # loop through all the pixels to find the extreme pixels
    for pixel in pixel_points:
        if pixel[1] > bottom_left[1] or (pixel[1] == bottom_left[1] and pixel[0] < bottom_left[0]):
            bottom_left = pixel
        if pixel[1] > bottom_right[1] or (pixel[1] == bottom_right[1] and pixel[0] > bottom_right[0]):
            bottom_right = pixel
        if pixel[1] < top_left[1] or (pixel[1] == top_left[1] and pixel[0] < top_left[0]):
            top_left = pixel
        if pixel[1] < top_right[1] or (pixel[1] == top_right[1] and pixel[0] > top_right[0]):
            top_right = pixel

    # return the extreme pixels
    return bottom_left, bottom_right, top_left, top_right


# In[214]:


bottom_left, bottom_right, top_left, top_right = find_extreme_pixels(patch16)
print(bottom_left,bottom_right,top_left,top_right)


# In[215]:


top_left = (max(0, min(top_left[0], 223)), max(0, min(top_left[1], 223)))
top_right = (max(0, min(top_right[0], 223)), max(0, min(top_right[1], 223)))
bottom_left = (max(0, min(bottom_left[0], 223)), max(0, min(bottom_left[1], 223)))
bottom_right = (max(0, min(bottom_right[0], 223)), max(0, min(bottom_right[1], 223)))


# In[216]:


# Determine the rectangle dimensions
width = bottom_right[0] - top_left[0]
height = bottom_right[1] - top_left[1]


# In[217]:


# Create a copy of the image to draw the rectangle on
image_with_rectangle = image.copy()


# In[218]:


# Extract the rectangular region from the original image
rectangular_region = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
plt.imshow(rectangular_region)


# In[219]:


# Draw the rectangle on the copied image
cv2.rectangle(image_with_rectangle, top_left, bottom_right, (0, 254, 0), thickness=2)


# In[220]:


plt.imshow(image_with_rectangle)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





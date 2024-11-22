#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import cv2

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2


# # guided-backpropagation algorithm to find the influence of each of the 5 neurons in last convolutional layer

# In[60]:


from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
    return tf.nn.relu(x), grad


class GuidedBackprop:
    def __init__(self,model, layerName=None):
        self.model = model
        self.layerName = layerName
        self.gbModel = self.build_guided_model()
        
        if self.layerName == None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply Guided Backpropagation")

    def build_guided_model(self):
        print(self.model.get_layer(self.layerName).output.shape)
        dum = self.model.get_layer(self.layerName).output[:, :, 9:,]
        #dum = tf.expand_dims(dum, axis=0)
        print(dum.shape)
        gbModel = Model(
            inputs = [self.model.inputs],
            outputs = [dum]
        )
        layer_dict = [layer for layer in gbModel.layers[1:] if hasattr(layer,"activation")]
        for layer in layer_dict:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guidedRelu
        
        return gbModel
    
    def guided_backprop(self, images, upsample_size, neuron_index = None):
        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            tape.watch(inputs)
            if neuron_index is None:
              outputs = self.gbModel(inputs)
            else:
              outputs = self.gbModel(inputs)[:,neuron_index]
            # outputs = self.gbModel(inputs)


            

        grads = tape.gradient(outputs, inputs)[0]

        saliency = cv2.resize(np.asarray(grads), upsample_size)

        return saliency

def show_BP(GuidedBP, im_ls, n=3):

    plt.subplots(figsize=(30, 10*n))
    k=1
    for i in range(n):
        img = image.load_img(im_ls[i])
        # Show original image
        plt.subplot(n,3,k)
        plt.imshow(img)
        plt.title("Filename: {}".format(im_ls[i]), fontsize=20)
        plt.axis("off")
        
        x = image.img_to_array(img)

        upsample_size = (x.shape[1],x.shape[0])
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)


        plt.subplot(n,3,k+1)
        gb = GuidedBP.guided_backprop(x, upsample_size)
        #print(gb)

        scal = MinMaxScaler()

        plt.imshow(gb)
        plt.title("Guided Backprop", fontsize=20)
        plt.axis("off")

        plt.subplot(n,3,k+2)
        gb[:, :, 0] = scal.fit_transform(gb[:, :, 0])
        gb[:, :, 1] = scal.fit_transform(gb[:, :, 1])
        gb[:, :, 2] = scal.fit_transform(gb[:, :, 2])

        plt.imshow(gb)
        plt.title("MinMaxScaled", fontsize=20)
        plt.axis("off")
        k += 3
    plt.show()


vgg = VGG19(include_top = False, weights= 'imagenet', input_shape=(224, 224, 3))
guidedBP = GuidedBackprop(model=vgg, layerName="block5_conv4")

img_paths = ['/content/drive/MyDrive/image_0001.jpg']

plt.figure(figsize=(15, 5))
for i, img_path in enumerate(img_paths):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    x = np.expand_dims(image.img_to_array(img), axis=0)
    x = preprocess_input(x)
    
    plt.subplot(len(img_paths), 3, i*3+1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    upsample_size = (img.size[1], img.size[0])
    plt.subplot(len(img_paths), 3, i*3+2)
    gb = guidedBP.guided_backprop(x, upsample_size, neuron_index = 0)
    
    # plt.imshow(0.6-gb)
    # plt.title("Guided Backprop")
    # plt.axis("off")
    gb_prop = gb
    guided_back_viz = np.dstack((
        gb_prop[:, :, 0],
        gb_prop[:, :, 1],
        gb_prop[:, :, 2],
    ))       
    guided_back_viz -= np.min(guided_back_viz)
    guided_back_viz /= guided_back_viz.max()

    imgplot = plt.imshow(guided_back_viz)
    plt.axis("off")
    plt.show()


    plt.subplot(len(img_paths), 3, i*3+3)
    gb[:, :, 0] = MinMaxScaler().fit_transform(gb[:, :, 0])
    gb[:, :, 1] = MinMaxScaler().fit_transform(gb[:, :, 1])
    gb[:, :, 2] = MinMaxScaler().fit_transform(gb[:, :, 2])
    plt.imshow(gb)
    plt.title("MinMaxScaled")
    plt.axis("off")
    
  
plt.show()


#print(x.shape)
image = plt.imread(img_path)
plt.imshow(image)

# plt.imshow(img_path)
# plt.show()


# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:






# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import os
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import random as rn
import itertools


# In[4]:


os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(29)  # For numpy numbers
rn.seed(29)   # For Python
tf.set_random_seed(29)    #For Tensorflow


# In[5]:


#Force tensorflow to use a single thread
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(sess)

train_path = "/home/ubuntu/Plant-Species-Recognition/PlantImages/train/"
test_path = "/home/ubuntu/Plant-Species-Recognition/PlantImages/test/"
valid_path = "/home/ubuntu/Plant-Species-Recognition/PlantImages/validation/"


# In[9]:


#Generates batches of tensor image data that images must be in
#F-f-d takes path and puts in batches of normalized data, one-hot encodes classes (class_mode argument)
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size = (224,224), batch_size = 10)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size = (224,224), batch_size = 5)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size = (224,224), batch_size = 5)


# In[ ]:


#### VGG 16 model ####
vgg16 = keras.applications.vgg16.VGG16()

#Create a sequential model with the VGG16 keras model functional API
#Iterate over every VGG 16 model and add to sequential type model
vgg16_model = Sequential()

for layer in vgg16.layers[:-1]:  #Removes the last layers with 1000 classes
    vgg16_model.add(layer)
    
for layer in vgg16_model.layers:
    layer.trainable = False #Freeze a layer so weights arent updated, use for finetuning just the end

#vgg16_model.add(Flatten())
vgg16_model.add(Dense(12, activation = 'softmax'))

vgg16_model.summary()


# In[ ]:


#### Training VGG 16 Model ####

vgg16_model.compile(Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

vgg16_model.fit_generator(train_batches, steps_per_epoch=4400, validation_data = valid_batches,
                    validation_steps=1550, epochs = 5, verbose = 1)

vgg16_model.save("VGG16_finetuned_model.h5")
# In[ ]:


# predictions = vgg16_model.predict_generator(test_batches, steps = 1600, verbose = 0)
# predictions


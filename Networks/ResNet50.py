
import numpy as np
from keras.applications import resnet50
from keras.layers.core import Dense, Flatten
from keras import backend as K
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, Input

### Reproducibility ###
import os
import random as rn
import tensorflow as tf
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(29)  # For numpy numbers
rn.seed(29)   # For Python
tf.set_random_seed(29)    #For Tensorflow

# #Force tensorflow to use a single thread
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(sess)

#Keras code goes after this


train_path = "/home/ubuntu/Plant-Species-Recognition/PlantImages/train/"
test_path = "/home/ubuntu/Plant-Species-Recognition/PlantImages/test/"
valid_path = "/home/ubuntu/Plant-Species-Recognition/PlantImages/validation/"

#Generates batches of tensor image data that images must be in
#F-f-d takes path and puts in batches of normalized data, one-hot encodes classes (class_mode argument)
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224, 224), batch_size=10)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224, 224), batch_size=5)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224, 224), batch_size=5)

image_input = Input(shape=(224, 224, 3))

resnet_model = resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None,
                                 input_shape=(224, 224, 3), pooling=None, classes=1000)

model = Sequential()

for layer in resnet_model.layers[:-1]:  # Removes the last layers with 1000 classes
    print(layer)
    model.add(layer)

# for layer in model.layers:
#     layer.trainable = False  # Freeze a layer so weights arent updated, use for finetuning just the end

# vgg16_model.add(Flatten())
#model.add(Dense(12, activation='softmax'))

#model.summary()

# last_layer = resnet_model.get_layer('avg_pool').output
# x = Flatten(name = 'flatten')(last_layer)
# # out = Dense(12, activation='softmax', name='output_layer')(x)
# # res_model = Model(inputs = image_input, outputs = out)
# # resnet_model.summary()

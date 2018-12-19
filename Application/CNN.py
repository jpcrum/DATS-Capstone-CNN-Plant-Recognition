import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random as rn
import tensorflow as tf
import pandas as pd

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import load_model

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import time


def load_images(file_path):
    image_filenames = []
    for root, dirs, files in os.walk(file_path):
        if len(files) > 0:
            for file in files:
                if(file[-3:] == "png" or file[-3:] == "PNG"):
                    image_filenames.append(str(root)+os.sep+str(file))

    df = pd.DataFrame(image_filenames,columns=['image_paths'])
    df.to_csv("predict_images.csv",index = False)

    return image_filenames




def preprocess_images(file_list):
    images = []
    for i, image in enumerate(file_list):
        im = cv2.imread(image)
        im = cv2.resize(im, (224, 224))
        norm_im = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        hsv = cv2.cvtColor(norm_im, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (40, 0, 0), (110, 255,255))
        imask = mask>0
        green = np.zeros_like(im, np.uint8)
        green[imask] = im[imask]
        medblur = cv2.medianBlur(green, 7)
        images.append(np.array(medblur))
        # if i % 10:
        #     cv2.imwrite('static/image{}.jpg'.format(i), medblur)
    return images




def predict_images(images):
    model = load_model("CNN_Models/CNN.h5")
    pred = []
    for image in images:
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        label = np.where(prediction[0] == max(prediction[0]))
        pred.append(int(label[0]))

    preds = np.zeros(12)
    for i in range(12):
        preds[i] = int(pred.count(i))
    preds = list(preds)
    return preds, pred, images



def torch_predict_images(images):

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.layer1 = nn.Sequential(
                #1 input for grayscale, # of feature maps,
                nn.Conv2d(3, 32, kernel_size=7, padding=3, stride=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=7, padding=3),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2))
            self.layer3 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=5, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2))
            self.layer4 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2))
            self.layer5 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2))
            self.fc1 = nn.Linear(7 * 7 * 32, 128)
            self.drop1 = nn.Dropout(0.2)
            self.fc2 = nn.Linear(128, 12)

        def forward(self, x):
            out = self.layer1(x.float())
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.layer5(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = self.drop1(out)
            out = self.fc2(out)
            return out

    cnn = CNN()
    cnn.load_state_dict(torch.load('CNN_Models/BigModel.pkl', map_location='cpu'))

    preds = []

    for image in images:
        #print(image)
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        #image1 = image.astype(np.float32)
        image = Variable(torch.from_numpy(image))
        outputs = cnn(image)
        p = outputs.data.cpu().numpy()
        a = np.exp(p[0])
        a_sum = sum(a)
        softmax = [b/a_sum for b in a]
        _, predicted = torch.max(outputs.data, 1)
        if (max(softmax) >= 0.7):
            preds.append(predicted.cpu().numpy()[0])
        else:
            preds.append(12)

    predictions = np.zeros(13)
    for i in range(13):
        predictions[i] = int(preds.count(i))
    counts = list(predictions)
    return preds, counts


def sort(preds, image_paths):
    classes = {'0':'Black-Grass', '1':'Charlock', '2':'Cleavers', '3':'Common Chickweed', '4':'Common wheat', '5':'Fat Hen', '6':'Loose Silky-bent',
               '7':'Maize', '8':'Scentless Mayweed', '9':'Shepherds Purse', '10':'Small-flowered Cranesbill', '11':'Sugar beet', '12':'Undecided'}
    pred = {}
    for i in range(len(image_paths)):
        pred[image_paths[i]] = preds[i]
    for image in pred.keys():
        paths = image.rsplit('\\', 1)
        prediction = str(pred[image])
        new_folder = classes[prediction]
        os.rename(image, paths[0] + '/' + new_folder + '/' + paths[1])

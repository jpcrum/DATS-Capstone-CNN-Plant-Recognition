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

    df = pd.DataFrame(image_filenames,columns=['image_paths','labels'])
    df.to_csv("predict_images.csv",index = False))

    return image_filenames




def preprocess_images(file_list):
    images = []
    i = 1
    for image in file_list:
        im = cv2.imread(image)
        im = cv2.resize(im, (100, 100))
        norm_im = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        hsv = cv2.cvtColor(norm_im, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (40, 0, 0), (100, 255,255))
        imask = mask>0
        green = np.zeros_like(im, np.uint8)
        green[imask] = im[imask]
        medblur = cv2.medianBlur(green, 13)
        images.append(np.array(medblur))
        cv2.imwrite('static/image{}.jpg'.format(i), medblur)
        i += 1
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
    model = load_model("CNN_Models/CNN.h5")

    class CustomDatasetFromImages(Dataset):
        def __init__(self, csv_path):
            self.data_info = pd.read_csv(csv_path)
            self.image_array = np.asarray(self.data_info.iloc[:, 0])
            self.data_len = len(self.data_info.index)

        def __getitem__(self, index):
            # Get image name from the pandas df
            single_image_name = self.image_array[index]
            img_as_img = cv2.imread(single_image_name)
            img_resized = cv2.resize(img_as_img, (100, 100))
            norm_im = cv2.normalize(img_resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            hsv = cv2.cvtColor(norm_im, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (40, 0, 0), (110, 255, 255))
            imask = mask > 0
            green = np.zeros_like(norm_im, np.uint8)
            green[imask] = norm_im[imask]
            medblur = cv2.medianBlur(green, 9)
            return (medblur)
        def __len__(self):
            return self.data_len

    if __name__ == '__main__':
        test_loader = CustomDatasetFromImages('predict_images.csv')

    print(test_loader)



def sort(preds, image_paths):
    classes = {'0':'Black-Grass', '1':'Charlock', '2':'Cleavers', '3':'Common Chickweed', '4':'Common wheat', '5':'Fat Hen', '6':'Loose Silky-bent',
               '7':'Maize', '8':'Scentless Mayweed', '9':'Shepherds Purse', '10':'Small-flowered Cranesbill', '11':'Sugar beet'}
    pred = {}
    for i in range(len(image_paths)):
        pred[image_paths[i]] = preds[i]
    for image in pred.keys():
        paths = image.rsplit('\\', 1)
        prediction = str(pred[image])
        new_folder = classes[prediction]
        # print(image)
        # print(paths[0])
        # print(new_folder)
        # print(paths[1])
        # print(paths[0] + new_folder + paths[1])
        os.rename(image, paths[0] + '/' + new_folder + '/' + paths[1])

import os
import random
import pandas as pd
import numpy as np
import cv2

classes = {'Black-grass': '0', 'Charlock': '1', 'Cleavers': '2', 'Common Chickweed': '3', 'Common wheat': '4',
           'Fat Hen': '5', 'Loose Silky-bent': '6', 'Maize': '7', 'Scentless Mayweed': '8', 'Shepherds Purse': '9',
           'Small-flowered Cranesbill': '10', 'Sugar beet': '11'}

def extract_paths(path):
    image_filenames = []
    for root, dirs, files in os.walk(path):
        if len(files) > 0:
            for file in files:
                image_filenames.append(str(root)+os.sep+str(file))
    return image_filenames

filenames_train = extract_paths('/home/ubuntu/PlantImageRecognition/ImageData/train/')
filenames_test = extract_paths('/home/ubuntu/PlantImageRecognition/ImageData/test/')

def make_csv(files, dataset):
    images_and_labels = []
    for image in files[1:]:
        # im = cv2.imread(image)
        # print(image)
        # if im.shape[1] >= 1000:
        #     print(im.shape[1])
        base = os.path.dirname(image).rsplit("/", 1)
        class_lab = base[1]
        label = classes[class_lab]
        image_and_label = [image, label]
        images_and_labels.append(image_and_label)

    df = pd.DataFrame(images_and_labels,columns=['image_paths','labels'])
    #print(df)
    df.to_csv('/home/ubuntu/PlantImageRecognition/ImageData/{}_images_and_labels.csv'.format(dataset), index = False)

make_csv(filenames_test, 'test')
make_csv(filenames_train, 'train')


print('a')
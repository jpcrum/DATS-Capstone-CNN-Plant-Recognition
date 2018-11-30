
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import keras

import random as rn
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(29)  # For numpy numbers
rn.seed(29)   # For Python


test_path = "/home/ubuntu/Plant-Species-Recognition/TryPics/"

image_filenames = []
for root, dirs, files in os.walk(test_path):
    if len(files) > 0:
        for file in files:
            if (file[-3:] == "png" or file[-3:] == "PNG"):
                image_filenames.append(str(root) + os.sep + str(file))
print(len(image_filenames))

image = cv2.imread("/home/ubuntu/Plant-Species-Recognition/TryPics/Cleaver.png")
image = cv2.resize(image, (100,100))
image = np.expand_dims(image, axis=0)

model = load_model("/home/ubuntu/Plant-Species-Recognition/MLTutorial/10layerCNN100x100-50epochs.h5")
#model.summary()

prediction = model.predict(image)


#predictions = model.predict(pred_batch, steps = 1, verbose = 1)


# preds = []
# for prediction in predictions:
#     label = np.where(prediction == max(prediction))[0][0] + 1
#     preds.append(label)


# labels = []
# for label in test_labels:
#     lab = np.where(label == max(label))[0][0]
#     labels.append(lab)
#
#
# classes = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common Wheat', 'Fat Hen', 'Loose Silky-Bend',
#            'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-Flowered Cranesbill', 'Sugar Beet']
#
# cm = confusion_matrix(labels, preds)
# print(cm)
#
# acc = accuracy_score(labels, preds, normalize=True, sample_weight=None)
# print(acc)
# #
# df_cm = pd.DataFrame(cm, index = classes, columns = classes)
# print(df_cm)
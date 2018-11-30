import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, random
random.seed(29)


def random_photo_per_class(path):
    random_images = []
    for root, dirs, files in os.walk(path):
        if root[-5:] != "train":
            image = random.choice(os.listdir("{}".format(root)))
            random_images.append(str(root)+os.sep+str(image))
    return random_images[1:]

rands = random_photo_per_class("/home/ubuntu/Plant-Species-Recognition/PlantImages/train/")


for image in rands:
    directory = os.path.dirname(image)
    species = directory.rsplit('\\', 1)[-1]
    #print(species)
    img = cv2.imread(image)

    #print(img.shape)

    plt.imshow(img)

    plt.show()


ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))

ts = ts.cumsum()
ts.plot()
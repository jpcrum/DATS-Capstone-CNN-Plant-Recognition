# -----------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import time
# -----------------------------------------------------------------------------------

image_size = 224


######### Image Data Generator ##############

class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path, transforms):
        # Read the csv file
        self.data_info = pd.read_csv(csv_path)
        # First column contains the image paths
        self.image_array = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_array = np.asarray(self.data_info.iloc[:, 1])
        # Calculate len
        self.data_len = len(self.data_info.index)
        self.transforms = transforms

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_array[index]
        img_as_img = cv2.imread(single_image_name)
        img_resized = cv2.resize(img_as_img, (image_size, image_size))

        #Preprocess the images
        norm_im = cv2.normalize(img_resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        hsv = cv2.cvtColor(norm_im, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (30, 0, 0), (110, 255, 255))
        imask = mask > 0
        green = np.zeros_like(norm_im, np.uint8)
        green[imask] = norm_im[imask]
        medblur = cv2.medianBlur(green, 9)

        single_image_label = self.label_array[index]
        # Return image and the label
        #return (img_resized, single_image_label)
        #return (norm_im, single_image_label)
        return (norm_im, single_image_label)


    def __len__(self):
        return self.data_len

if __name__ == '__main__':
    train_loader = CustomDatasetFromImages('/home/ubuntu/PlantImageRecognition/ImageData/train_images_and_labels.csv', transforms = transforms)
    test_loader = CustomDatasetFromImages('/home/ubuntu/PlantImageRecognition/ImageData/test_images_and_labels.csv', transforms = transforms)

#transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

num_epochs = 30
batch_size = 100
learning_rate = 0.001

#Load the data
train = DataLoader(train_loader, batch_size = batch_size, shuffle=True)
test = DataLoader(test_loader, batch_size = batch_size, shuffle=True)

#Iterate over data
train_iter = iter(train)
test_iter = iter(test)

images, labels = train_iter.next()

print('images shape on batch size = {}'.format(images.size()))
print('labels shape on batch size = {}'.format(labels.size()))

# -----------------------------------------------------------------------------------
# CNN Model (2 conv layer)
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
        self.softmax = nn.Softmax()

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
        #out = self.softmax(out)
        return out
# -----------------------------------------------------------------------------------
cnn = CNN()
cnn.cuda()
# -----------------------------------------------------------------------------------
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
# -----------------------------------------------------------------------------------
# Train the Model

losses = []

start_time = time.time()
for epoch in range(num_epochs):
    print ("Starting Epoch {}".format(epoch + 1))
    train_iter = iter(train)
    #i = 0
    for i, (images, labels) in enumerate(train_iter):
        # images = torch.from_numpy(images)
        # labels = torch.from_numpy(np.ndarray(labels))
        images_fixed = [image.permute(2,0,1) for image in images]
        images = torch.stack(images_fixed)

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #i += 1

        if i % 100 == 0:
            print('Epoch {}/{}, Iter {}/{}, Loss: {}'.format(epoch + 1, num_epochs, i, int(train_loader.data_len / batch_size), loss.item()))

        losses.append(loss.item())

    print("Epoch Done")

print("--- %s seconds ---" % (time.time() - start_time))
# -----------------------------------------------------------------------------------
# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0

im_labels = []
im_preds = []

for i, (images, labels) in enumerate(test_iter):
    images_fixed = [image.permute(2, 0, 1) for image in images]
    images = torch.stack(images_fixed)
    images = Variable(images).cuda()
    outputs = cnn(images)
    if i % 10 == 0:
        print("Preicting {}/{}...".format(i, len(test_iter)))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

    im_labels.append(labels.cpu().numpy())
    im_preds.append(predicted.cpu().numpy())

print('Done Predicting!')
# -----------------------------------------------------------------------------------
print('Test Accuracy of the model on the test images: {}'.format(100 * correct / total))
# -----------------------------------------------------------------------------------
# Save the Trained Model

preds = [x for pred in im_preds for x in pred]
labels = [x for label in im_labels for x in label]

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

cm = confusion_matrix(labels, preds)
print(cm)

acc = accuracy_score(labels, preds, sample_weight=None)
print("Accuracy: " + str(acc))

f1 = f1_score(labels, preds, average='macro')
print("F1: " + str(f1))
recall = recall_score(labels, preds, average='macro')
print("Recall: " + str(recall))
precision = precision_score(labels, preds, average='macro')
print("Precision: " + str(precision))

torch.save(cnn.state_dict(), '/home/ubuntu/PlantImageRecognition/Models/BigModeNorm.pkl')

weights = []
for i in range(32):
    kernel = cnn._modules['layer1']._modules['0'].weight.data[i][0].cpu().numpy()
    weights.append(kernel)

flat_weights = []
for weight in weights:
    flat_weight = [x for row in weight for x in row]
    flat_weights.append(flat_weight)

weights_df = pd.DataFrame(flat_weights)
weights_df.to_csv('/home/ubuntu/PlantImageRecognition/Metrics/BigModelNorm.csv')

print('Done')

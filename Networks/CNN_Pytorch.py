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

image_size = 100

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

        norm_im = cv2.normalize(img_resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        hsv = cv2.cvtColor(norm_im, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (40, 0, 0), (100, 255, 255))
        imask = mask > 0
        green = np.zeros_like(norm_im, np.uint8)
        green[imask] = norm_im[imask]
        medblur = cv2.medianBlur(green, 13)

        #img_final = np.expand_dims(medblur, 0)
        single_image_label = self.label_array[index]
        # Return image and the label
        return (medblur, single_image_label)


    def __len__(self):
        return self.data_len

if __name__ == '__main__':
    train_loader = CustomDatasetFromImages('/home/ubuntu/PlantImageRecognition/ImageData/train_images_and_labels.csv', transforms = transforms)
    test_loader = CustomDatasetFromImages('/home/ubuntu/PlantImageRecognition/ImageData/test_images_and_labels.csv', transforms = transforms)

#transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

num_epochs = 20
batch_size = 100
learning_rate = 0.001

train = DataLoader(train_loader, batch_size = batch_size, shuffle=True)
test = DataLoader(test_loader, batch_size = batch_size, shuffle=True)

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
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(6 * 6 * 32, 128)
        self.fc2 = nn.Linear(128, 16)

    def forward(self, x):
        out = self.layer1(x.float())
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
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
    i = 0
    for images, labels in train_iter:
        # images = torch.from_numpy(images)
        # labels = torch.from_numpy(np.ndarray(labels))
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        i += 1

        if (i) % 1 == 0:
            print('Epoch {}/{}, Iter {}/{}, Loss: {}'.format(epoch + 1, num_epochs, i, (train_loader.data_len / batch_size), loss.item()))

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

for images, labels in test_iter:
    images = Variable(images).cuda()
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

    im_labels.append(labels.cpu().numpy())
    im_preds.append(predicted.cpu().numpy())
# -----------------------------------------------------------------------------------
print('Test Accuracy of the model on the test images: {}'.format(100 * correct / total))
# -----------------------------------------------------------------------------------
# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn-224-4ConvBlocks-00001lr-7733kernel-Adam-100Epochs.pkl')
#torch.save(cnn.state_dict(), 'cnn-wholeimage-224-4ConvBlocks-00001lr-11kernel-Adam.pkl".pkl')

preds = [x for pred in im_preds for x in pred]
labels = [x for label in im_labels for x in label]

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

cm = confusion_matrix(labels, preds)
print(cm)

acc = accuracy_score(labels, preds, sample_weight=None)
print("Accuracy: " + acc)

f1 = f1_score(labels, preds, average='macro')
print("F1: " + f1)
recall = recall_score(labels, preds, average='macro')
print("Recall: " + recall)
precision = precision_score(labels, preds, average='macro')
print("Precision: " + precision)

plt.figure(figsize=(10, 6.18))

ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

plt.xlim(1, num_epochs)
plt.ylim(-0.01, np.max(losses)+0.5)

plt.xticks(range(1, num_epochs + 1))

for y in np.arange(0, np.max(losses) + 0.1, 1):
    plt.plot(range(1, 6), [y] * len(range(1, num_epochs + 1)), "--", lw=0.5, color="black", alpha=0.3)

plt.tick_params(axis="both", which="both", bottom="off", top="off",
                labelbottom="on", left="off", right="off", labelleft="on")

ax.plot(losses, lw=1.5, color='#E45555')

plt.title('{} vs {}'.format(metric, parameter), fontsize=14, y=1.03)
plt.xlabel('Epoch', fontsize=12, labelpad=10)
plt.ylabel('{}'.format(metric), fontsize=12, labelpad=10)
plt.show()


print(a)

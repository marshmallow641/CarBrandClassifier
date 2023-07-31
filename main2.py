import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from PIL import Image
import os
import pandas as pd
import numpy as np
import warnings
from torchsummary import summary

from helper_script import resize_training_images, generate_annotations_csv

# Image Dataset
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, width=256, height=256, channels=3):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        self.width = width
        self.height = height
        self.channels = channels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path_1 = os.path.join(str(self.img_dir), str(self.img_labels.iloc[idx, 2]))
        img_path_2 = os.path.join(str(img_path_1), str(self.img_labels.iloc[idx, 0]))

        img = read_image(img_path_2)
        img = img / img.amax(keepdim=True)
        lbl = self.img_labels.iloc[idx, 1]

        '''if lbl == 0:
            lbl = torch.tensor(data=[0.0,-1.0,-1.0], dtype=torch.float)
        elif lbl == 1:
            lbl = torch.tensor(data=[-1.0,0.0,-1.0], dtype=torch.float)
        elif lbl == 2:
            lbl = torch.tensor(data=[-1.0,-1.0,0.0], dtype=torch.float)'''

        if self.transform:
            img = self.transform(img)

        return img, lbl

 # Sets the device (CUDA GPU) and the default tensor datatype for the script/session
DEVICE = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
#torch.set_default_device(DEVICE)
torch.cuda.reset_peak_memory_stats(f"{DEVICE}")
torch.set_default_dtype(torch.float)

# Neural Network Hyperparameters
EPOCHS = 20
LEARNING_RATE = 0.002
BATCH_SIZE = 64
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
CHANNELS = 3
NUMBER_OF_WORKERS = 1 # Needs to be set to 1 if running script directly in the model class/GPU to take advantage of GPU's speed

# Variables for preparing and retrieving training and testing data
annotation_headers = ["File Name", "Class ID", "Category", "File Path", "Original Image Size"]

saved_model_path = "model-state-dicts"
train_data_path = "D:\\Deep Learning Datasets\\car_brands"
test_data_path = "D:\\Deep Learning Datasets\\car_brands_test"
resized_train_image_path = "resized images train"
resized_test_image_path = "resized images test"

train_annotations_file = "cars_annotation.csv"
test_annotations_file = "cars_annotation_test.csv"

test_margin = 0.05

# Create an annotations file (.csv file), resize the training images to be the same size, and load the data into a DataLoader object
generate_annotations_csv(filename=train_annotations_file, dir=train_data_path, headers=annotation_headers, path_code=2)
resize_training_images(train_data_path, resized_train_image_path, train_annotations_file, IMAGE_WIDTH, IMAGE_HEIGHT)

# Create an annotations file (.csv file), resize the testing images to be the same size, and load the data into a DataLoader object
generate_annotations_csv(filename=test_annotations_file, dir=test_data_path, headers=annotation_headers, path_code=2)
resize_training_images(img_dir=test_data_path, new_dir=resized_test_image_path, annotation_file=test_annotations_file, width=IMAGE_WIDTH, height=IMAGE_HEIGHT)

class MyImageNetwork(nn.Module):
    def __init__(self, channels=1):
        super().__init__()

        self.input_channels = channels
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")

        self.conv2d_relu_stack = nn.Sequential(nn.Conv2d(in_channels=self.input_channels, out_channels=32, kernel_size=5, padding=1, stride=2),
                                               nn.ReLU(),
                                               nn.Dropout(0.2),
                                               nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=1, stride=2),
                                               nn.ReLU(),
                                               nn.Dropout(0.2),
                                               nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                                               nn.ReLU(),
                                               nn.MaxPool2d(2),
                                               nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                                               nn.ReLU(),
                                               nn.Dropout(0.2),
                                               nn.Conv2d(in_channels=256, out_channels=128, kernel_size=2),
                                               nn.ReLU(),
                                               nn.MaxPool2d(2),
                                               nn.Flatten(),
                                               nn.Linear(in_features=28800, out_features=500),
                                               nn.ReLU(),
                                               nn.Dropout(0.2),
                                               nn.Linear(in_features=500, out_features=200),
                                               nn.Dropout(0.2),
                                               nn.ReLU(),
                                               nn.Linear(in_features=200, out_features=41))

    def forward(self, x):
        logits = self.conv2d_relu_stack(x)

        return logits

    def fit(self):
        my_image_dataset = CustomImageDataset(annotations_file=train_annotations_file, img_dir=resized_train_image_path, transform=None)
        train_image_dataloader = DataLoader(my_image_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUMBER_OF_WORKERS, pin_memory=True, pin_memory_device=DEVICE)

        test_data = CustomImageDataset(annotations_file=test_annotations_file, img_dir=resized_test_image_path)
        test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=NUMBER_OF_WORKERS, pin_memory=True, pin_memory_device=DEVICE)
        ############################################ Training the model ################################################################################################
        losses: list = []
        loss = nn.CrossEntropyLoss()
        l: object

        model = MyImageNetwork(channels=CHANNELS).to(DEVICE)
        summary(model, (CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT), BATCH_SIZE)

        optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE, momentum=0.9)

        if not os.path.exists(saved_model_path):
            os.makedirs(saved_model_path)

        # CNN Training Loop
        for epoch in range(EPOCHS):
            # Display current epoch
            print("============")
            print(f"| Epoch {epoch + 1}  |")
            print("============")

            optimizer.zero_grad()

            # Loop over all batches at each epoch
            for i in range(my_image_dataset.__len__() // BATCH_SIZE):
                print(f"Batch: {i + 1} out of {(my_image_dataset.__len__() // BATCH_SIZE)}, ", end="")

                training_images, training_labels = next(iter(train_image_dataloader))

                training_images_mean = torch.mean(training_images)
                training_images_std = torch.std(training_images)

                training_images = torch.tensor(data=((training_images - training_images_mean) / training_images_std),
                                               dtype=torch.float, device=DEVICE)
                training_labels = torch.tensor(data=training_labels, dtype=torch.float, device=DEVICE).to(DEVICE)

                prediction = model(training_images).to(DEVICE)
                prediction = torch.argmax(prediction, dim=1)
                prediction = torch.tensor(data=prediction, dtype=torch.float, device=DEVICE, requires_grad=True)



                l = loss(prediction, training_labels).to(DEVICE)
                l.backward()


            optimizer.step()
            losses.append(l)

            for i in range(len(losses)):
                print(f"Epoch {i + 1} CNN Loss: {losses[i]}")

            print("\n")

            torch.save(model.state_dict(), f"{saved_model_path}/training-my_image_model{epoch + 1}_loss{l}.pth")

        ##################################### Testing the saved models ###################################################################################################
        correct = 0
        total = 0
        epoch = 0

        models = os.listdir(saved_model_path)

        # CNN Test Loop
        with torch.no_grad():
            for model_ in models:
                model.load_state_dict(torch.load(f"{saved_model_path}/{model_}"))
                model.to(DEVICE)
                model.eval()

                for data in test_dataloader:
                    test_images, test_labels = data
                    test_images = torch.tensor(data=test_images, device=DEVICE)
                    test_labels = torch.tensor(data=test_labels, device=DEVICE)

                    outputs = model(test_images)

                    _, predicted = torch.max(outputs.data, 1)
                    total += test_labels.size(0)
                    correct += (predicted - test_labels < test_margin).sum().item()

                print(
                    f"Model ({model_}) Accuracy ({test_data.__len__()} instances): {correct} out of {total} predictions were correct (within a {test_margin * 100}% margin) - {int((correct / total) * 100)}% accuracy")

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    model = MyImageNetwork().to(DEVICE)
    model.fit()

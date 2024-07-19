from brush import Brush
import matplotlib.pyplot as plt
from network import CharacterRecognizer
import numpy as np
import tkinter as tk
import torch
import torch.nn as nn
import torch.optim as optim   
from torch.utils.data import DataLoader 
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

BATCH_SIZE = 64
PATH_TO_SAVE = './model/character_recognizer.pth'

device = None
if torch.cuda.is_available():
  device = torch.device('cuda')
else: 
  device = torch.device('cpu')

	 
training_data = datasets.EMNIST(
  root="data",
  split="balanced",
  train=True,
  download=True,
  transform=torchvision.transforms.Compose([
    lambda img: torchvision.transforms.functional.rotate(img, -90),
    lambda img: torchvision.transforms.functional.hflip(img),
    torchvision.transforms.ToTensor()
  ])
)

test_data = datasets.EMNIST(
  root="data",
  split="balanced",
  train=False,
  download=True,
  transform=torchvision.transforms.Compose([
    lambda img: torchvision.transforms.functional.rotate(img, -90),
    lambda img: torchvision.transforms.functional.hflip(img),
    torchvision.transforms.ToTensor()
  ])
)

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

model = CharacterRecognizer().to(device)
criterion = nn.CrossEntropyLoss() # performance of a model. will look into later
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # scholastic gradient descent

epochs = 50
for epoch in range(epochs):

  for inputs, labels in train_dataloader:

    inputs = inputs.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()

    output = model(inputs)
    loss = criterion(output, labels)

    loss.backward()
    optimizer.step()

  accuracy = 0
  count = 1

  for inputs, labels in test_dataloader:

    inputs = inputs.to(device)
    labels = labels.to(device)
  
    output = model(inputs)
    accuracy += (torch.argmax(output, 1) == labels).float().sum()
    count += len(labels)

  accuracy /= count
  print("Epoch %d: model accuracy %.2f%%" % (epoch, accuracy * 100))

print("finish training, should save model")
torch.save(model.state_dict(), PATH_TO_SAVE)



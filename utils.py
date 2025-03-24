#!/usr/bin/env python
# coding: utf-8

# Import dependencies

# In[ ]:


import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Preprocess and load EMNIST dataset

# In[3]:


# Define transformation for data preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure it's grayscale
    transforms.Resize((28, 28)),                  # Resize to 28x28 pixels
    transforms.ToTensor(),                        # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))          # Normalize between -1 and 1
])

# Load dataset (using EMNIST Balanced for both letters and digits)
train_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, transform=transform, download=True)
test_dataset = datasets.EMNIST(root='./data', split='balanced', train=False, transform=transform, download=True)

# DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


# Define logic behind CNN

# In[4]:


# Define a CNN for OCR
class OCRCNN(nn.Module):
    def __init__(self):
        super(OCRCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Adjust for 28x28 image size after conv and pooling layers
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 47)  # 47 classes for the EMNIST Balanced dataset (letters + digits)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Convolution -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(-1, 64 * 7 * 7)  # Flatten the output for fully connected layers
        
        x = F.relu(self.fc1(x))     # Fully connected layer 1 with ReLU
        x = F.relu(self.fc2(x))     # Fully connected layer 2 with ReLU
        x = self.fc3(x)             # Output layer (logits, no activation here)
        return x


# Initialize model and begin training using CUDA for GPU acceleration.

# In[8]:


# Initialize model, loss function, and optimizer

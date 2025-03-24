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



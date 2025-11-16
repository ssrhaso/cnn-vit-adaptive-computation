"""LOAD CIFAR10 DATASET"""
import torch
import torchvision
import numpy as np
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10


print("Downloading CIFAR10 dataset")
transform = transforms.Compose([                                # Define image transformations
    transforms.ToTensor(),                                      # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))      # Normalize images with mean and std deviation for each channel (R, G, B)
])

testset = CIFAR10(root='./data', train=False, download=True, transform=transform)       # Download and load the test set
print("CIFAR10 dataset downloaded successfully, size : ", {len(testset)}, )
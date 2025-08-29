import torch
import torch.nn as nn
import numpy as np

import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

#torch.manual_seed(17)
#np.random.seed(17)

class RGBHSVDataset(Dataset):
    """Custom Dataset for RGB-HSV conversion"""
    def __init__(self, rgb_data, hsv_data):
        self.rgb_data = torch.FloatTensor(rgb_data)
        self.hsv_data = torch.FloatTensor(hsv_data)
        
    def __len__(self):
        return len(self.rgb_data)
    
    def __getitem__(self, idx):
        return self.rgb_data[idx], self.hsv_data[idx]
    
class HSVRGBDataset(Dataset):
    """Custom Dataset for HSV-RGB conversion"""
    def __init__(self, hsv_data, rgb_data):
        self.hsv_data = torch.FloatTensor(hsv_data)
        self.rgb_data = torch.FloatTensor(rgb_data)
        
    def __len__(self):
        return len(self.hsv_data)
    
    def __getitem__(self, idx):
        return self.hsv_data[idx], self.rgb_data[idx]

class RGBtoHSV(nn.Module):
    def __init__(self):
        super(RGBtoHSV, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            # No activation on final layer as HSV has different ranges
        )
    
    def forward(self, x):
        return self.model(x)
    
class MNISTNetwork(nn.Module):
    def __init__(self):
        super(MNISTNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x
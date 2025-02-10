
import copy
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision import transforms, models, utils

#Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

class CTNet(nn.Module):
    """Model for big data. ResNet18 then 3D conv then FC."""
    def __init__(self, n_outputs):
        super(CTNet, self).__init__()        

        resnet = models.resnet18()
        self.features = nn.Sequential(*(list(resnet.children())[:-2]))
        
        #conv input torch.Size([1,134,512,14,14])
        self.reducingconvs = nn.Sequential(
            nn.Conv3d(134, 64, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(),
            
            nn.Conv3d(64, 32, kernel_size = (3,3,3), stride=(3,1,1), padding=0),
            nn.ReLU(),
            
            nn.Conv3d(32, 16, kernel_size = (3,2,2), stride=(3,2,2), padding=0),
            nn.ReLU())
        
        self.classifier = nn.Sequential(
            nn.Linear(16*18*5*5, 128), #7200
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 96), 
            nn.ReLU(True),
            nn.Dropout(0.5),
            
            nn.Linear(96, n_outputs))
        
    def forward(self, x):
        shape = list(x.size())
        #example shape: [1,134,3,420,420]
        #example shape: [2,134,3,420,420]
        batch_size = int(shape[0])
        x = x.view(batch_size*134,3,420,420)
        features = self.features(x)
        del x
        x = features
        x = x.view(batch_size,134,512,14,14)
        x = self.reducingconvs(x)
        #output is shape [batch_size, 16, 18, 5, 5]
        x = x.view(batch_size, 16*18*5*5)
        x = self.classifier(x)
        return x
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

class CTNetNoConv(nn.Module):
    """Model for big data. ResNet18 then 3D conv then FC."""
    def __init__(self, n_outputs):
        super(CTNetNoConv, self).__init__()        

        resnet = models.resnet18()
        self.features = nn.Sequential(*(list(resnet.children())[:-1]))
        
        self.classifier = nn.Sequential(

            nn.Linear(134*512, 16*18*5*5), #7200
            nn.ReLU(True),
            nn.Dropout(0.5),

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
        # print(x.shape)
        features = self.features(x)
        del x
        x = features
        # print(x.shape)
        x = x.view(batch_size,134*512)
        #output is shape [batch_size, 16, 18, 5, 5]
        # x = x.view(batch_size, 16*18*5*5)
        x = self.classifier(x)
        return x
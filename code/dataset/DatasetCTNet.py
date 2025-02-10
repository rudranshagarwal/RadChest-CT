import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.optim as optim
from torch.utils.data import DataLoader as dataloader_normal, TensorDataset
from utils import prepare_ctvol_2019_10_dataset
import nibabel as nib
from sklearn.neighbors import NearestNeighbors
import numpy as np


class DatasetCTNet(Dataset):
    
   
    def __init__(self, annotations_file, image_dir, lungmask_dir,labels, exclude, split, data_augment):
        self.img_labels = pd.read_csv(annotations_file)
        self.labels = []
        self.image_name = []
        self.image_paths = []
        self.lungmask_paths = []
        self.split = split
        self.pixel_bounds = [-1000, 200]
        self.num_channels = 3
        self.crop_type = 'single'
        if self.split == 'train':
            self.data_augment = data_augment
        else:
            self.data_augment = False
        for set in os.listdir(image_dir):
            set_path = os.path.join(image_dir, set)
            set_path_lungmask = os.path.join(lungmask_dir, set)
            for image in os.listdir(set_path):
                image = image.split(".")[0]
                image_path = os.path.join(set_path, image)
                lungmask_path = os.path.join(set_path_lungmask, image + '.nii')
                self.image_name.append(image)
                self.image_paths.append(image_path)
                self.lungmask_paths.append(lungmask_path)
                # if(feature_path in exclude):
                #     continue
                # self.feature_paths.append(feature_path)

                label_names = [label + '*lung' for label in labels]                
                # for image in self.img_labels['NoteAcc_DEID'].unique():  
                #     label_values = self.img_labels[self.img_labels['NoteAcc_DEID'] == image][label_names].values[0]
                #     self.labels.append(torch.tensor(label_values))
                self.labels.append(torch.tensor(self.img_labels[self.img_labels['NoteAcc_DEID'] == image][label_names].values[0]))
                # self.labels.append(torch.tensor(self.img_labels[self.img_labels['NoteAcc_DEID'] == image][['reticulation*lung']].values[0]))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # feature = torch.load(self.feature_paths[idx])
        # print(os.path.join(self.image_paths[idx]) + '.npz')
        lungmask = nib.load(self.lungmask_paths[idx]).get_fdata()
        ctvol = np.load(os.path.join(self.image_paths[idx]) + '.npz')['ct']
        label = self.labels[idx]
        data = prepare_ctvol_2019_10_dataset(ctvol, self.pixel_bounds, self.data_augment, self.num_channels, self.crop_type, lungmask)
        return data, label, self.image_name[idx]
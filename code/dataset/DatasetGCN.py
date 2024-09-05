import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.optim as optim
from torch.utils.data import DataLoader as dataloader_normal, TensorDataset
import nibabel as nib
from sklearn.neighbors import NearestNeighbors
import numpy as np


class CustomDataset(Dataset):
    
    def get_edges(self,centroids):
        

        k_neighbors = self.k
        knn_model = NearestNeighbors(n_neighbors=k_neighbors)

        knn_model.fit(centroids)

        distances, indices = knn_model.kneighbors(centroids)

        indices = indices[:, :]
        edges = [[],[]]
        for i in range(420):
            for j in range(self.k):
                edges[0].append(i)
                edges[1].append(indices[i][j])
        return edges
    def __init__(self, annotations_file, feature_dir, slic_dir, lungmask_dir,centroids_dir, k, labels, exclude):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = feature_dir
        self.feature_paths = []
        self.labels = []
        self.slic_paths = []
        self.lungmask_paths = []
        self.k = k
        self.centroid_paths = []
        self.image_name = []
        
        for set in os.listdir(feature_dir):
            set_path = os.path.join(feature_dir, set)
            set_path_slic = os.path.join(slic_dir,set)
            set_path_lungmask = os.path.join(lungmask_dir, set)
            set_path_centroids = os.path.join(centroids_dir, set)
            for feature in os.listdir(set_path):
                image = feature.split(".")[0]
                self.image_name.append(image)
                feature_path = os.path.join(set_path, feature)
                slic_path = os.path.join(set_path_slic, image + '.nii')
                lungmask_path = os.path.join(set_path_lungmask, image + '.nii')
                centroids_path = os.path.join(set_path_centroids, image + '.npy')
                if(feature_path in exclude):
                    continue
                self.feature_paths.append(feature_path)
                self.slic_paths.append(slic_path)
                self.lungmask_paths.append(lungmask_path)
                self.centroid_paths.append(centroids_path)
                label_names = [label + '*lung' for label in labels]                
                # for image in self.img_labels['NoteAcc_DEID'].unique():  
                #     label_values = self.img_labels[self.img_labels['NoteAcc_DEID'] == image][label_names].values[0]
                #     self.labels.append(torch.tensor(label_values))
                self.labels.append(torch.tensor(self.img_labels[self.img_labels['NoteAcc_DEID'] == image][label_names].values[0]))
                # self.labels.append(torch.tensor(self.img_labels[self.img_labels['NoteAcc_DEID'] == image][['reticulation*lung']].values[0]))

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        feature = torch.load(self.feature_paths[idx])
        label = self.labels[idx]
        
        # slic = nib.load(self.slic_paths[idx]).get_fdata()
        
        centroids = np.load(self.centroid_paths[idx])
        
        # lungmask = nib.load(self.lungmask_paths[idx]).get_fdata()
        edges = self.get_edges(centroids)
        
        return feature, label, torch.tensor(edges), self.image_name[idx]
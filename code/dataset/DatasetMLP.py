from torch.utils.data import Dataset
import os
import torch
import pandas as pd

class DatasetMLP(Dataset):
    def __init__(self, annotations_file, feature_dir,labels, exclude):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = feature_dir
        self.feature_paths = []
        self.labels = []
        self.slic_paths = []
        self.lungmask_paths = []
        self.centroid_paths = []
        self.image_name = []
        
        for set in os.listdir(feature_dir):
            set_path = os.path.join(feature_dir, set)
            
            for feature in os.listdir(set_path):
                image = feature.split(".")[0]
                self.image_name.append(image)
                feature_path = os.path.join(set_path, feature)
                
                if(feature_path in exclude):
                    continue
                self.feature_paths.append(feature_path)
                
                label_names = [label + '*lung' for label in labels]                
                # for image in self.img_labels['NoteAcc_DEID'].unique():  
                #     label_values = self.img_labels[self.img_labels['NoteAcc_DEID'] == image][label_names].values[0]
                #     self.labels.append(torch.tensor(label_values))
                self.labels.append(torch.tensor(self.img_labels[self.img_labels['NoteAcc_DEID'] == image][label_names].values[0]))
                # self.labels.append(torch.tensor(self.img_labels[self.img_labels['NoteAcc_DEID'] == image][['reticulation*lung']].values[0]))

    def __len__(self):
        return len(self.feature_paths)

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        feature = torch.load(self.feature_paths[idx])
        label = self.labels[idx]
        
        return torch.mean(feature, dim=0), label, self.image_name


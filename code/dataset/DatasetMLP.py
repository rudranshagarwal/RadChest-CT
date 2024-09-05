from torch.utils.data import Dataset
import os
import torch
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, annotations_file, feature_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = feature_dir
        self.feature_paths = []
        self.labels = []
        problem = ["/scratch/features/train/set22/trn24016.pth",
"/scratch/features/train/set12/trn12411.pth",
"/scratch/features/train/set9/trn09320.pth",
"/scratch/features/train/set18/trn19397.pth",
"/scratch/features/train/set7/trn07778.pth",
"/scratch/features/train/set14/trn15124.pth",
"/scratch/features/train/set14/trn15109.pth",
"/scratch/features/train/set10/trn10310.pth",
"/scratch/features/train/set13/trn14334.pth", 
"/scratch/features/test/set4/tst35252.pth",
"/scratch/features/test/set1/tst29356.pth",
"/scratch/features/test/set3/tst34126.pth"
]
        for set in os.listdir(feature_dir):
            set_path = os.path.join(feature_dir, set)
            for feature in os.listdir(set_path):
                feature_path = os.path.join(set_path, feature)
                if(feature_path in problem):
                    continue
                self.feature_paths.append(feature_path)
                
                image = feature.split(".")[0]
                self.labels.append(torch.tensor(self.img_labels[self.img_labels['NoteAcc_DEID'] == image][['nodule*lung', 'opacity*lung', 'atelectasis*lung', 'consolidation*lung', 'mass*lung', 'pneumothorax*lung']].values[0]))
                # self.labels.append(torch.tensor(self.img_labels[self.img_labels['NoteAcc_DEID'] == image][['opacity*lung']].values[0]))

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        feature = torch.load(self.feature_paths[idx])
        label = self.labels[idx]
        
        return feature, label


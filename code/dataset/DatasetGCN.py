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
import networkx as nx
from scipy.linalg import eigh

class CustomDataset(Dataset):
    def get_edges(self,centroids, connectivity, image):
        
        
        k_neighbors = 420
        knn_model = NearestNeighbors(n_neighbors=k_neighbors)

        knn_model.fit(centroids)

        distances, indices = knn_model.kneighbors(centroids)

        indices = indices[:, :]
        edges = [[],[]]
        edge_tuples = []
        for i in range(420):
            count = 0
            j = 0
            if j == 420:
                print(i)
            while count < self.k:
                if(self.connectivity == 'lung'):
                    x = -1
                    y = -1
                    z = indices[i][j]
                    if connectivity[i] == 1 or connectivity[i] == 2:
                        x = 1
                    else:
                        x = 2
                    if connectivity[z] == 1 or connectivity[z] == 2:
                        y = 1
                    else:
                        y = 2
                    if( x != y):
                        j+=1
                        continue
                
                elif(self.connectivity == 'lobe'):
                    if(j >= 420):
                        j = i
                    elif(connectivity[i] != connectivity[indices[i][j]]):
                        j+=1
                        continue
                    
                        
                    
                edges[0].append(indices[i][j])
                edges[1].append(i)
                edge_tuples.append((i,indices[i][j]))
                count +=1 
                j+=1
        
        return edges, edge_tuples
    def __init__(self, annotations_file, feature_dir, slic_dir, lungmask_dir,centroids_dir, k, labels, exclude, connectivity_dir, connectivity):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = feature_dir
        self.connectivity_dir = connectivity_dir
        self.connectivity = connectivity
        self.feature_paths = []
        self.labels = []
        self.slic_paths = []
        self.lungmask_paths = []
        self.k = k
        self.centroid_paths = []
        self.image_name = []
        self.connectivity_paths = []
        
        for set in os.listdir(feature_dir):
            set_path = os.path.join(feature_dir, set)
            set_path_slic = os.path.join(slic_dir,set)
            set_path_lungmask = os.path.join(lungmask_dir, set)
            set_path_centroids = os.path.join(centroids_dir, set)
            set_path_connectivity = os.path.join(connectivity_dir, set)
            for feature in os.listdir(set_path):
                image = feature.split(".")[0]
                
                feature_path = os.path.join(set_path, feature)
                slic_path = os.path.join(set_path_slic, image + '.nii')
                lungmask_path = os.path.join(set_path_lungmask, image + '.nii')
                centroids_path = os.path.join(set_path_centroids, image + '.npy')
                connectivity_path = os.path.join(set_path_connectivity, image + '.pt')
                if(feature_path in exclude):
                    continue
                
                label_names = [label + '*lung' for label in labels] 
                currlabels = torch.tensor(self.img_labels[self.img_labels['NoteAcc_DEID'] == image][label_names].values[0], dtype=torch.float32)
                # print(image)
                # if(image != 'tst29333' and (image[:3] != 'trn') and (image[:3] != 'val')):
                #     continue
                
                # if(image == 'tst30676'):
                #     print(int(currlabels.tolist()[7]) != 1, int(currlabels.tolist()[-2]) == 1)
                # if(int(currlabels.tolist()[-2]) == 1):
                #     continue
                # if(int(currlabels.tolist()[7]) != 1):
                #     continue
                # if((image[:3] == 'tst') and image != 'tst35936'):
                #     continue
                # if(currlabels[2] != 1):
                #     continue
                self.image_name.append(image)
                self.feature_paths.append(feature_path)
                self.slic_paths.append(slic_path)
                self.lungmask_paths.append(lungmask_path)
                self.centroid_paths.append(centroids_path)
                self.connectivity_paths.append(connectivity_path)
                label_names = [label + '*lung' for label in labels]                
                # for image in self.img_labels['NoteAcc_DEID'].unique():  
                #     label_values = self.img_labels[self.img_labels['NoteAcc_DEID'] == image][label_names].values[0]
                #     self.labels.append(torch.tensor(label_values))
                self.labels.append(torch.tensor(self.img_labels[self.img_labels['NoteAcc_DEID'] == image][label_names].values[0], dtype=torch.float32))
                # self.labels.append(torch.tensor(self.img_labels[self.img_labels['NoteAcc_DEID'] == image][['reticulation*lung']].values[0]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = torch.load(self.feature_paths[idx], weights_only=True)
    
        label = self.labels[idx]
        # print(feature.dtype)
        # slic = nib.load(self.slic_paths[idx]).get_fdata()
        
        centroids = np.load(self.centroid_paths[idx])
        connectivity = torch.load(self.connectivity_paths[idx], weights_only=False)
        # lungmask = nib.load(self.lungmask_paths[idx]).get_fdata()
        # _, counts = np.unique(connectivity, return_counts=True)

        # for i, count in enumerate(counts):
        #     if count < 10:
        #         print(self.image_name[idx], i + 1)
        edges, edge_tuples = self.get_edges(centroids, connectivity, self.image_name[idx])  

        # G = nx.Graph() 

        # G.add_nodes_from(range(420))  
        # G.add_edges_from(edge_tuples)
        
        # laplacian_matrix = nx.laplacian_matrix(G).toarray()

        # eigenvalues, eigenvectors = eigh(laplacian_matrix)

        # laplacian = torch.stack((torch.tensor(eigenvalues[:20]).repeat(420).reshape(420,20), torch.tensor(eigenvectors[:, :20])), dim=-1).float()
        return feature, label, torch.tensor(edges), self.image_name[idx], torch.tensor(connectivity)

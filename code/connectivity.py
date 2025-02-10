import numpy as np
import nibabel as nib
import torch

import sys

lungmask_file = sys.argv[1]
slic_file = sys.argv[2]
output_file = sys.argv[3]

mask_5 = nib.load(lungmask_file).get_fdata()  # 0-5 labels
mask_420 = nib.load(slic_file).get_fdata()  # 0-240 labels

results = []
labels = np.unique(mask_420)
regions = []
for label in labels[1:]: 
    label_voxels = (mask_420 == label)
    
    if np.sum(label_voxels) == 0:
        continue
    
    regions_of_label = mask_5[label_voxels]
    
    unique, counts = np.unique(regions_of_label, return_counts=True)
    
    majority_region = unique[np.argmax(counts)]
    if(majority_region == 0):
         majority_region = unique[np.argsort(counts)[-2]]
    
    majority_count = np.max(counts)
    
    total_count = np.sum(counts)
    majority_percentage = (majority_count / total_count) * 100
    
    results.append((label, majority_region, majority_percentage))

    regions.append(majority_region)


torch.save(regions, output_file)
# import matplotlib.pyplot as plt
# import numpy as np
# import SimpleITK as sitk

# from skimage.segmentation import slic, quickshift, felzenszwalb
# from skimage.segmentation import mark_boundaries
# from skimage.measure import regionprops
# from skimage.util import img_as_float
# from skimage import io, exposure, util

# # from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# import pickle
# import glob
# import sys
# import nrrd

# # file_start = int(sys.argv[1])
# # file_end = int(sys.argv[2])

# # files = glob.glob("ctrs/*.mha")

# def read_mha_file(filepath):
#     """Read and load volume"""
#     # Read file
#     scan_sitk = sitk.ReadImage(filepath, sitk.sitkFloat32) 
#     # Get raw data
#     scan = sitk.GetArrayFromImage(scan_sitk)
#     return scan

# def read_mask_file(filepath):
#     """Read and load volume"""
#     # Read file
#     mask_sitk = sitk.ReadImage(filepath, sitk.sitkInt16) 
#     # Get raw data
#     mask = sitk.GetArrayFromImage(mask_sitk)
#     return mask

# def normalize(volume):
#     """Normalize the volume"""
#     min = -1350
#     max = 150
#     volume[volume < min] = min
#     volume[volume > max] = max
#     volume = (volume - min) / (max - min)
#     volume = volume.astype("float32")
#     return volume

# def process_scan(path):
#     """Read and resize volume"""
#     # Read scan
#     volume = read_mha_file(path)
#     # Normalize
#     volume = normalize(volume)
#     return volume

# # print("\nSupervoxelation for files from: {} to: {}\n".format(file_start, file_end))

# # for k in range(file_start, file_end):
    

# scanpath = "/home/rudransh/Desktop/CVIT/Lungs/output_image.nii"
# maskpath = "/home/rudransh/Desktop/CVIT/Lungs/test.nii"

# print("\nProcessing file: {} and mask: {}\n".format(scanpath, maskpath))

# scan = process_scan(scanpath)
# mask = read_mask_file(maskpath)

# num_supervoxels = 42

# num_query = 44
# num_extracted = 43

# compactness = 0.9

# count = 0

# while num_extracted != num_supervoxels:

#     count += 1

#     if count > 1:
#         print("No solution in 6 iterations for {}.".format("test"))
#         break

#     segments_slic = slic(scan, compactness=compactness, sigma=0, enforce_connectivity=True,
#                 n_segments=num_query, start_label=1, mask=mask, channel_axis=None)

#     indices = np.unique(segments_slic)
#     num_extracted = len(indices) 

#     print("Number of query supervoxels: {} Number of extracted superpixels: {}".format(num_query, num_extracted))

#     if num_extracted > num_supervoxels:
#         num_query -=1
#     elif num_extracted < num_supervoxels:
#         num_query +=1
#     else:
#         if num_query == num_extracted:
#             compactness -=0.1
#         else:
#             num_query +=1

# slic_segmentation = np.swapaxes(segments_slic, 0, 2)
# print(np.unique(slic_segmentation))
# nrrd.write('/home/rudransh/Desktop/CVIT/Lungs/' + str("test_slic") + '.nrrd', slic_segmentation)
            
#     # np.save('/ssd_scratch/cvit/chocolite/data/slic_small/' + str(file), segments_slic)

#     # centroids = []

#     # centroids_1 = []
#     # centroids_2 = []

#     # regions = regionprops(segments_slic)
    
#     # for props in regions:
#     #     cx, cy, cz = props.centroid
#     #     cx = int(cx)
#     #     cy = int(cy)
#     #     cz = int(cz)
#     #     # print(cx, cy, cz)

#     #     centroids.append([cx, cy, cz])

#     #     # print(mask[cx, cy, cz] == 1)

#     #     if mask[cx, cy, cz] == 1:
#     #         centroids_1.append([cx, cy, cz])
#     #     elif mask[cx, cy, cz] == 2:
#     #         centroids_2.append([cx, cy, cz])
                
#     # centroids = np.asarray(centroids)

#     # from sklearn.neighbors import kneighbors_graph

#     # A = kneighbors_graph(centroids, 3, mode='distance', include_self=False)
#     # A.toarray()

#     # import networkx as nx

#     # G = nx.from_numpy_matrix(np.matrix(A.toarray()), create_using=nx.DiGraph)

#     # layout = nx.spring_layout(G)

import numpy as np
import SimpleITK as sitk
import nibabel as nib
from skimage.segmentation import slic
import argparse

# import nrrd

# from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def read_mha_file(filepath):
    """Read and load volume"""
    # Read file
    scan_sitk = sitk.ReadImage(filepath, sitk.sitkFloat32) 
    # Get raw data
    scan = sitk.GetArrayFromImage(scan_sitk)
    return scan

def read_mask_file(filepath):
    """Read and load volume"""
    # Read file
    mask_sitk = sitk.ReadImage(filepath, sitk.sitkInt16) 
    # Get raw data
    mask = sitk.GetArrayFromImage(mask_sitk)
    return mask

def normalize(volume):
    """Normalize the volume"""
    min = -1350
    max = 150
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_mha_file(path)
    # Normalize
    volume = normalize(volume)
    return volume

# print("\nSupervoxelation for files from: {} to: {}\n".format(file_start, file_end))

# for k in range(file_start, file_end):
    
parser = argparse.ArgumentParser(description='Process an input file.')
parser.add_argument('scan_file', type=str)
parser.add_argument('mask_file', type=str)
parser.add_argument('output_file', type=str)

args = parser.parse_args()
scanpath = args.scan_file
maskpath = args.mask_file
outputpath = args.output_file

print("\nProcessing file: {} and mask: {}\n".format(scanpath, maskpath))

scan = process_scan(scanpath)
mask = read_mask_file(maskpath)


def super_voxelation(scan, mask, segments, start):

    num_supervoxels = segments + 1

    num_query = segments + 1
    num_extracted = segments - 1

    compactness = 0.9
    segments_slic = []
    count = 0

    while num_extracted != num_supervoxels:

        count += 1

        if count > 5:
            print("No solution in 6 iterations for {}.".format("test"))
            break

        segments_slic = slic(scan, compactness=compactness, sigma=0, enforce_connectivity=True,
                    n_segments=num_query, start_label=start, mask=mask, channel_axis=None)

        indices = np.unique(segments_slic)
        num_extracted = len(indices) 

        print("Number of query supervoxels: {} Number of extracted superpixels: {}".format(num_query, num_extracted))

        if num_extracted > num_supervoxels:
            num_query -=1
        elif num_extracted < num_supervoxels:
            num_query +=1
        else:
            if num_query == num_extracted:
                compactness -=0.1
            else:
                num_query +=1
    # slic_segmentation = np.swapaxes(segments_slic, 0, 2)
    slic_segmentation = segments_slic
    return slic_segmentation


mask = read_mask_file(maskpath)
combined_slic_segmentation = np.zeros(mask.shape)
new_mask = np.where((mask == 1) | (mask == 2), 1, 0)
slic_segmentation = super_voxelation(scan,new_mask, 200, 1)
combined_slic_segmentation += slic_segmentation
new_mask = np.where((mask == 3) | (mask == 4) | (mask==5), 1, 0)
slic_segmentation = super_voxelation(scan,new_mask, 220,1)
combined_slic_segmentation += np.where((slic_segmentation !=0), slic_segmentation+200, 0)
image = nib.load(scanpath)
combined_slic_segmentation = np.swapaxes(combined_slic_segmentation, 0,2)
new_image = nib.Nifti1Image(combined_slic_segmentation, image.affine, image.header)
nib.save(new_image, outputpath + '.nii')


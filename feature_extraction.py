import torch
from torchvision.models import resnet50, resnet18
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import DataLoader,Dataset
import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from torchvision.models import ResNet50_Weights
from torchvision import transforms
import argparse

m = resnet50()

return_nodes = {
    'layer4': 'layer4',
    'avgpool': 'avgpool'
}

body = create_feature_extractor(m, return_nodes=return_nodes)

parser = argparse.ArgumentParser(description='Process an input file.')
parser.add_argument('set_path', type=str, help='The input file to be processed')
parser.add_argument('slic_path', type=str, help='The input file to be processed')
parser.add_argument('output_path', type=str, help='The output file to be saved at')
args = parser.parse_args()
images = os.listdir(args.set_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
body = body.to(device)
body.eval()
for image in images:
    print('Processing: ', image)

    image_path = os.path.join(args.set_path, image)
    slic_path = os.path.join(args.slic_path, image.split('.')[0] + '.nii')
    image_data = np.load(image_path)['ct']
    try:
        slic_data = nib.load(slic_path).get_fdata()
    except:
        print("Image slic not found: ", slic_path)
        continue
    
    def get_bounding_boxes(labels, data):
        unique_labels = np.unique(labels)
        bounding_boxes = {}

        for label in unique_labels:
            min_coords = [None] * 3
            max_coords = [None] * 3
            label_mask = (data == label)
            for dim in range(3):
                min_idx = np.argmax(np.any(label_mask, axis=tuple(i for i in range(3) if i != dim)))
                min_coords[dim] = min_idx

                if(dim == 0):
                    max_idx = label_mask.shape[dim] - np.argmax(np.any(label_mask[::-1], axis=tuple(i for i in range(3) if i != dim)))
                elif(dim == 1):
                    max_idx = label_mask.shape[dim] - np.argmax(np.any(label_mask[:, ::-1], axis=tuple(i for i in range(3) if i != dim)))
                elif(dim == 2):
                    max_idx = label_mask.shape[dim] - np.argmax(np.any(label_mask[:, :, ::-1], axis=tuple(i for i in range(3) if i != dim)))
                max_coords[dim] = max_idx

            bounding_boxes[label] = (min_coords, max_coords)
        
        return bounding_boxes

    def extract_cubes(image, labels, slic):
        bounding_boxes = get_bounding_boxes(labels, slic)
        cubes = {}
        
        for label, (min_coords, max_coords) in bounding_boxes.items():
            cube_image = image[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]
            cube_labels = slic[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]
            cube_image = np.where(cube_labels == label, cube_image, 0)
            cube_labels = np.where(cube_labels == label, cube_labels, 0)
            cubes[label] = (cube_image, cube_labels)
        
        return cubes

    labels = np.unique(slic_data)[1:]
    cubes = extract_cubes(image_data, np.unique(slic_data)[1:], slic_data)
    print('Cubes Extracted')
    features = torch.empty((0,2048)).to(device)
    preprocess = transforms.Compose([
            transforms.Resize(256),                      # Resize the shorter side to 256 pixels
            transforms.CenterCrop(224),                  # Crop the center 224x224 pixels
            transforms.Normalize(mean=[0.485, 0.456, 0.406],   # Normalize using ImageNet stats
                                std=[0.229, 0.224, 0.225])
        ])
    
    with torch.no_grad():
        for j in labels:
            slices = torch.empty((0, 3, 224, 224))
            segment = torch.tensor(cubes[j][0], dtype=torch.float32)
            nslices = segment.shape[0]
            if(nslices % 3 == 1):
                segment = torch.cat((segment,segment[-1:]), dim = 0)
                segment = torch.cat((segment,segment[-1:]), dim = 0)
            elif(nslices % 3 == 2):
                segment = torch.cat((segment,segment[-1:]), dim = 0)
            slices = torch.empty((0, 3, 224, 224))
            for i in range(0,segment.shape[0], 3):
                transformed = preprocess(segment[i:i+3,:,:])
                slices = torch.cat((slices,transformed.unsqueeze(0)), dim=0)
            feature = torch.mean(body(slices.to(device))['avgpool'], dim=0).reshape(2048)
            features = torch.cat((features, feature.unsqueeze(0)))
    torch.save(features.to('cpu'), os.path.join(args.output_path,image.split('.')[0]+ '.pth'))
    print(image + ' Features extracted')

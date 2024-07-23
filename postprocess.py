import numpy as np
from scipy.ndimage import convolve
import nibabel as nib
import SimpleITK as sitk
import argparse

parser = argparse.ArgumentParser(description='Process an input file.')
parser.add_argument('input_file', type=str, help='The input file to be processed')
parser.add_argument('output_file', type=str, help='The output file to be saved at')
args = parser.parse_args()

image = sitk.ReadImage(args.input_file)
labels = list(sitk.GetArrayViewFromImage(image).flatten())
labels = list(set(labels)) 
labels.remove(0)  

radius = 10
kernel = sitk.sitkBall
closing_filter = sitk.BinaryMorphologicalClosingImageFilter()
closing_filter.SetKernelRadius(radius)
closing_filter.SetKernelType(kernel)

closed_image = sitk.Image(image.GetSize(), sitk.sitkUInt8)
closed_image.CopyInformation(image)
for label in labels:
    binary_image = sitk.BinaryThreshold(image, lowerThreshold=float(label), upperThreshold=float(label), insideValue=1, outsideValue=0)
    closed_binary_image = closing_filter.Execute(binary_image)
    closed_image = sitk.Maximum(closed_image, sitk.Cast(closed_binary_image, sitk.sitkUInt8) * label)

sitk.WriteImage(closed_image, args.output_file)

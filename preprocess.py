import numpy as np
import nibabel as nib
import argparse

parser = argparse.ArgumentParser(description='Process an input file.')
parser.add_argument('input_file', type=str, help='The input file to be processed')
parser.add_argument('output_file', type=str,help='The output file to be saved at')
args = parser.parse_args()
nifti_filename = args.output_file
affine = np.array([[0.,0., -1.,0.], [0.,-1.,0.,0.],[-1.,0.,0.,0.],[0.,0.,0.,1.]])
data_array = np.load(args.input_file)['ct']
nifti_img = nib.Nifti1Image(data_array, affine)
nib.save(nifti_img,nifti_filename)
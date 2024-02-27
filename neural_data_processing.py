#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from glob import glob
import os 

import numpy as np
import pandas as pd 

import nibabel as nib
import nilearn
from nilearn.maskers import NiftiMasker
from tqdm import tqdm


### HOW TO LOAD BETA IMAGES and mask them to hippocampus ROI ###

        # change the directories as you need :) 
#roi_path = "/home/mariachiara/Escritorio/fMRI_Analyses/ROIs"
betas_path = "classifier_betas_encoding"
general_path = "Behavior"  # behavioral data


#hc_mask = nib.load(os.path.join(roi_path, "BilatHippocampalAAL.nii"))
hc_mask = nib.load("BilatHippocampalAAL.nii")

print(hc_mask)
######################### 
people = [2, 3, 4, 7, 8, 9, 12, 13, 15, 17, 18, 19, 20, 21, 23, 24, 26, 27, 28, 30, 31, 32, 33] # 6 has been discarded
runs = [1, 2, 3, 4]

all_parts = []

for j in tqdm(people, desc="processing participants"):
    
    whole_participant_dictionary = []
    
    for f in runs: 
        
        beta_images_folder = os.path.join(betas_path, f"betas_{j}_run{f}")
        image_paths = list(glob(f"{beta_images_folder}/beta_*.nii")) # gets a list of all the betas
                                                                     # of each run 
        image_paths.sort()    
        
        beta_images = [nib.load(image) for image in image_paths] # loads the images. They're at whole brain level!
        # -> The result fo nib.load is an image object, consisting of a 3D array. If you want to see the whole array, 
            # use the {image}.get_fdata() function
            
        # Masking the data to the hippocampus ROI:
            # 1. For each participant, load the mean functional image and use it to "calibrate" the ROI
        mean_participant_mask = nib.load(f"Mean_Func_Participant/sub-{j}_Mean_Func.nii.gz")
        reaffined_mask = nilearn.image.resample_img(hc_mask, target_affine=mean_participant_mask.affine, target_shape = mean_participant_mask.get_fdata().shape, interpolation='nearest', fill_value=0)
        
            # 2. Create a masker object with the ROI 
        masker = NiftiMasker(mask_img = reaffined_mask)
        
            # 3. Mask the data
        masked_betas = [masker.fit_transform(image)[0] for image in beta_images]
        # -> Returns a list with 5 arrays, one for each border/landmark 
        # containing the values of each voxel in the hippocampus.
        # This HC mask has 904 voxels, so each array has 904 values.
        
        
        ### Now that we have the HC activation of the run, let's get the behavioral information

        if len(str(j)) == 1:
            participant = pd.read_csv(os.path.join(general_path, f"sub-0{j}", f"p{j}_Run{f}.csv"))
        else:
            participant = pd.read_csv(os.path.join(general_path, f"sub-{j}", f"p{j}_Run{f}.csv"))

        # Print columns to check the column names

        shape = np.unique(participant["Shape"])[0]
        condition = np.unique(participant["Condition"])[0] # 1 when participants perform square shape first,
                                                        # 2 when participants perform distorted shape first
                                                        
        # What I usually do with this is create a dictionary, so I can store and easily access all the info I want
        # and then get a list of dictionaries
        run_dictionary = {"Functional": masked_betas, "Participant": j, "Run": f, "Shape": shape, "Condition": condition}
        whole_participant_dictionary.append(run_dictionary)

        
    all_parts.append(whole_participant_dictionary)

import pickle

with open('processed_data.pickle', 'wb') as f:
    pickle.dump(all_parts, f)

with open('processed_data.pickle', 'rb') as f:
    all_parts = pickle.load(f)
print(f"Here the results: {all_parts}")

print(10*"-", "PREPROCESSING DATA FINISHED SUCCESSFULLY AND SAVED", 10*"-")


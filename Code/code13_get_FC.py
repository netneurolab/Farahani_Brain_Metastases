"""
###############################################################################

Calculate the FC per HCP aging subject and save the concatinated FC matrix.

###############################################################################
"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import globals
import scipy.io
import numpy as np
import nibabel as nib
from scipy.stats import zscore
from neuromaps.images import load_data
from neuromaps.images import dlabel_to_gifti
from netneurotools.datasets import fetch_schaefer2018
from globals import path_FC, path_medialwall, path_results, path_atlas, path_templates

#------------------------------------------------------------------------------
# Get subject list
#------------------------------------------------------------------------------

path_info_sub = '/media/afarahani/Expansion1/Aging_HCP/' # this is where functional data is located

# Collect subfolders
subject_ids = sorted([
    d for d in os.listdir(path_info_sub)
    if os.path.isdir(os.path.join(path_info_sub, d)) and not d.startswith('.')])
num_subjects = len(subject_ids)

#------------------------------------------------------------------------------
# Get the parcellation to use - cortex-cerebellum
#------------------------------------------------------------------------------

template_paths = {'cortex_subcortex': os.path.join(path_templates, 'cortex_subcortex.dscalar.nii')}
templates = {key: nib.cifti2.load(path) for key, path in template_paths.items()}
template = templates['cortex_subcortex']

atlas_cerebellum = nib.load(path_results + 'Cerebellum_cifti_2mm.dscalar.nii').get_fdata()[0,globals.num_cort_vertices_noMW:]
atlas_cortex = nib.load(path_atlas + '/Cortex-Subcortex/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S1.dlabel.nii').get_fdata()[0,:globals.num_cort_vertices_noMW]
atlas_cortex = atlas_cortex - 16
atlas_cortex[atlas_cortex == -16] = 0
atlas_cortex[atlas_cortex != 0 ]  = atlas_cortex[atlas_cortex != 0 ] + 34
# np.unique(atlas_cortex) --> 0, 35, ... 434

atlas_combined = np.zeros((1, globals.num_vertices_voxels))

atlas_combined[0,:globals.num_cort_vertices_noMW] = atlas_cortex
atlas_combined[0,globals.num_cort_vertices_noMW:] = atlas_cerebellum
# len(np.unique(atlas_combined)) --> 433

n_labels = len(np.unique(atlas_combined)[1:])
labels = np.unique(atlas_combined)[1:]

#------------------------------------------------------------------------------
# Load subject specific data and process them
#------------------------------------------------------------------------------

schaefer = fetch_schaefer2018('fslr32k')[f"{globals.nnodes_Schaefer}Parcels7Networks"]
atlas = load_data(dlabel_to_gifti(schaefer))
mask_medial_wall = scipy.io.loadmat(path_medialwall + 'fs_LR_32k_medial_mask.mat')['medial_mask']
mask_medial_wall = mask_medial_wall.astype(np.float32)
atlas_cifti = atlas[(mask_medial_wall.flatten()) == 1]

#------------------------------------------------------------------------------
# Process subjects
#------------------------------------------------------------------------------

ca = 0 # counter
for s, subid in enumerate(subject_ids):
    print(s)
    if s >= 0:

        path_timeseries = f'/media/afarahani/Expansion1/Aging_HCP/{subid}/MNINonLinear/Results/'
        files = [
            'rfMRI_REST1_AP/rfMRI_REST1_AP_Atlas_MSMAll_hp0_clean.dtseries.nii',
            'rfMRI_REST1_PA/rfMRI_REST1_PA_Atlas_MSMAll_hp0_clean.dtseries.nii',
            'rfMRI_REST2_AP/rfMRI_REST2_AP_Atlas_MSMAll_hp0_clean.dtseries.nii',
            'rfMRI_REST2_PA/rfMRI_REST2_PA_Atlas_MSMAll_hp0_clean.dtseries.nii']

        data_list = []
        for file in files:
            inputFile = os.path.join(path_timeseries, file)
            if os.path.exists(inputFile):
                try:
                    img = nib.cifti2.load(inputFile)
                    data = img.get_fdata()[:, :]
                    data = data - np.mean(data, axis = 0)
                    data_list.append(data)
                except Exception as e:
                                print(f"Error {inputFile}: {e}")
            else:
                print(f"Missing: {inputFile}")
        if not data_list:
            print(f"No data found for subject {subid}")
            continue

        data_zc_list = []
        for data in data_list:
            data_zc = np.zeros((n_labels, np.shape(data)[0]))
            c= 0 
            for n in labels:
                data_zc[c, :] = np.nanmean(data[:, atlas_combined.flatten() == n], axis = 1)
                c = c+ 1
            data_zc = zscore(data_zc, axis = 1)
            data_zc_list.append(data_zc)
        data = np.concatenate(data_zc_list, axis = 1)

        # Compute functional connectome per individual
        FC = np.corrcoef(data)
        np.save(path_FC + 'FC_' + subid + '.npy', FC)

    ca += 1

#------------------------------------------------------------------------------
# Save the concatinated version ordered based on the clean csv file
#------------------------------------------------------------------------------

FC_all = np.zeros((432, 432, num_subjects))
for s, subid in enumerate(subject_ids):
    FC_all[:,:, s] = np.load(path_FC + 'FC_' + subid + '.npy')

# Save the functional connectome of all individuals concatinated
np.save(path_results + 'FC_all.npy', FC_all)

#------------------------------------------------------------------------------
# END
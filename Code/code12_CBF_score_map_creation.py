"""
###############################################################################

Create perfusion data using only HCP-A dataset
Explained-variance: [0.47683615 0.0255444 0.01333499 0.00835079 0.0078367]

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
from pathlib import Path
from scipy.stats import zscore
from sklearn.decomposition import PCA
from functions import save_as_dscalar_and_npy
from globals import path_medialwall, path_results, path_atlas, path_templates, path_perfusion

#------------------------------------------------------------------------------
# Load medial wall
#------------------------------------------------------------------------------

mask_medial_wall = scipy.io.loadmat(path_medialwall + 'fs_LR_32k_medial_mask.mat')['medial_mask']
mask_medial_wall = mask_medial_wall.astype(np.float32)

#------------------------------------------------------------------------------
# Helper function
#------------------------------------------------------------------------------

def load_gifti_data(fp: Path) -> np.ndarray:
    """Load a GIFTI as a 1D float array of vertices."""
    img = nib.load(str(fp))
    data = img.agg_data()
    data = np.asarray(data).squeeze()
    if data.ndim != 1:
        data = np.asarray(data)[:, 0]
    return data.astype(float)

#------------------------------------------------------------------------------
# Create a combined CIFTI atlas
#------------------------------------------------------------------------------

template_paths = {'cortex_subcortex': os.path.join(path_templates, 'cortex_subcortex.dscalar.nii')}
templates = {key: nib.cifti2.load(path) for key, path in template_paths.items()}
template = templates['cortex_subcortex']

atlas_cerebellum = nib.load(path_results + 'Cerebellum_cifti_2mm.dscalar.nii').get_fdata()[0,globals.num_cort_vertices_noMW:]
atlas_cortex = nib.load(path_atlas + '/Cortex-Subcortex/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S1.dlabel.nii').get_fdata()[0,:globals.num_cort_vertices_noMW]
atlas_cortex = atlas_cortex - 16
atlas_cortex[atlas_cortex == -16] = 0
atlas_cortex[atlas_cortex != 0 ] = atlas_cortex[atlas_cortex != 0 ] + 32
# np.unique(atlas_cortex) --> 0, 33, ... 433

atlas_combined = np.zeros((1, globals.num_vertices_voxels))

atlas_combined[0,:globals.num_cort_vertices_noMW] = atlas_cortex
atlas_combined[0,globals.num_cort_vertices_noMW:] = atlas_cerebellum
# len(np.unique(atlas_combined)) --> 433

# Save the CIFTI file
new_img = nib.Cifti2Image(atlas_combined,
                          header=template.header,
                          nifti_header=template.nifti_header)
new_img.to_filename(os.path.join(path_results, 'cerebellum_cortex.dscalar.nii'))

#------------------------------------------------------------------------------
# Load data (CBF)
#------------------------------------------------------------------------------

att_data = np.load(path_perfusion + 'perfusion_all_vertex.npy')

data_vertexwise_pca = zscore(att_data, axis = 0)
num_components = 5
pca = PCA(n_components = num_components, random_state = 1234)
scores_data = pca.fit_transform(data_vertexwise_pca)
loadings_data = (pca.components_.T * np.sqrt(pca.explained_variance_))

# Save components
for i in range(num_components):
    type_data =  'cortex_subcortex'
    save_as_dscalar_and_npy(scores_data[:, i], type_data, path_results, f'CBF_PCscore_{i}')

#------------------------------------------------------------------------------
# Parcellate the map
#------------------------------------------------------------------------------

n_labels = len(np.unique(atlas_combined)[1:])
labels = np.unique(atlas_combined)[1:]

parcellated_data = np.zeros((n_labels, 1))
c = 0
temp = scores_data[:, 0]
for i in labels:
    parcellated_data[c,:] = np.nanmean(temp[atlas_combined.flatten() == i], axis = 0)
    c = c + 1
np.save(path_results + 'parcellated_CBF.npy', parcellated_data)

#------------------------------------------------------------------------------
# END
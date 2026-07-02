#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import globals
import scipy.io
import numpy as np
import nibabel as nib
import seaborn as sns
from pathlib import Path
from scipy.stats import zscore
import matplotlib.pyplot as plt
from nilearn.connectome import ConnectivityMeasure
from globals import path_medialwall, path_results, path_atlas, path_templates, path_perfusion

#------------------------------------------------------------------------------
# Load medialwall
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
# Create a combined cifti atlas
#------------------------------------------------------------------------------

template_paths = {'cortex_subcortex': os.path.join(path_templates, 'cortex_subcortex.dscalar.nii')}
templates = {key: nib.cifti2.load(path) for key, path in template_paths.items()}
template = templates['cortex_subcortex']

atlas_cerebellum = nib.load(path_results + 'Cerebellum_cifti_2mm.dscalar.nii').get_fdata()[0,globals.num_cort_vertices_noMW:]
atlas_cortex = nib.load(path_atlas + '/Cortex-Subcortex/Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S1.dlabel.nii').get_fdata()[0,:globals.num_cort_vertices_noMW]
atlas_cortex = atlas_cortex - 16
atlas_cortex[atlas_cortex == -16] = 0
atlas_cortex[atlas_cortex != 0 ] = atlas_cortex[atlas_cortex != 0] + 32
# np.unique(atlas_cortex) --> 0, 33, ... 432

atlas_combined = np.zeros((1, globals.num_vertices_voxels))

atlas_combined[0,:globals.num_cort_vertices_noMW] = atlas_cortex
atlas_combined[0,globals.num_cort_vertices_noMW:] = atlas_cerebellum
# len(np.unique(atlas_combined)) --> 433

# Save the CIFTI file
new_img = nib.Cifti2Image(atlas_combined,
                          header=template.header,
                          nifti_header=template.nifti_header)
new_img.to_filename(os.path.join(path_results, "cerebellum_cortex.dscalar.nii"))

#------------------------------------------------------------------------------
# Load perfusion data
#------------------------------------------------------------------------------

all_data = np.load(path_perfusion + 'perfusion_all_vertex.npy')
num_subjects = len(all_data.T)

#------------------------------------------------------------------------------
# Parcellate the perfusion co-variance matrix
#------------------------------------------------------------------------------

n_labels = len(np.unique(atlas_combined)[1:])
labels = np.unique(atlas_combined)[1:]

parcellated_data = np.zeros((n_labels, num_subjects))
c = 0

for i in labels:
    parcellated_data[c,:] = np.nanmean(all_data[atlas_combined .flatten() == i,:], axis = 0)
    c = c + 1

#------------------------------------------------------------------------------
# Calculate similarity
#------------------------------------------------------------------------------
# Type 1
normalized_sub = zscore(parcellated_data, axis = 0)
data_normalized = zscore(normalized_sub, axis = 1)

correlation_measure = ConnectivityMeasure(kind = 'covariance')
corr_similarity = correlation_measure.fit_transform([data_normalized.T])[0]

plt.figure(figsize=(5,5))
sns.heatmap(corr_similarity, cmap = 'coolwarm', vmin = -1, vmax = 1)
plt.show()
np.save(path_results + 'blood_similarity_432.npy', corr_similarity)

# Type 2
data_normalized = zscore(parcellated_data, axis = 1)

correlation_measure = ConnectivityMeasure(kind = 'covariance')
corr_similarity = correlation_measure.fit_transform([data_normalized.T])[0]

plt.figure(figsize=(5,5))
sns.heatmap(corr_similarity, cmap = 'coolwarm',  vmin = 0.65, vmax = 1)
plt.show()
np.save(path_results + 'zscored_blood_similarity_432.npy', corr_similarity)

#------------------------------------------------------------------------------
# END
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from globals import path_results
from scipy.spatial.distance import cdist
from brainspace.null_models import MoranRandomization

#------------------------------------------------------------------------------
# Helper function
#------------------------------------------------------------------------------

def load_nifti(atlas_path):
    """
    Load nifti data
    """
    img =nib.load(atlas_path)
    return img, img.get_fdata()

#------------------------------------------------------------------------------
# Load original parcellated data
#------------------------------------------------------------------------------

cancer_name = 'Breast'
cancer_cereb = np.load(path_results + cancer_name + '_34_cerebellum.npy')[:32]
cancer_ctx = np.load(path_results + cancer_name + '_400_surf.npy').flatten()

data = np.concatenate((cancer_cereb, cancer_ctx), axis = 0)

#------------------------------------------------------------------------------
# Load atlas (combined cerebellum and schaefer-400)
#------------------------------------------------------------------------------

atlas_path = os.path.join(path_results, 'cerebellum_cortex_atlas_09.nii.gz')
atlas = nib.load(atlas_path).get_fdata()

atlas = atlas - 2
atlas[atlas<=0] = 0

# Calculate distance matrix
labels = np.unique(atlas)[1:]
com = np.zeros((len(labels), 3))
for i, lab in enumerate(labels):
    temp = (atlas == lab)
    com[i, :] = ndi.center_of_mass(temp)

distmat = cdist(com, com)
W = np.zeros_like(distmat)
mask = distmat > 0
W[mask] = 1.0 / distmat[mask]
W = (W + W.T) / 2
np.fill_diagonal(W, 0)

# Calculate moran nulls
msr = MoranRandomization(
    n_rep = 10000,
    tol = 1e-6,
    procedure ='singleton',
    random_state = 0)

msr.fit(W)
rand_data = msr.randomize(data)

# Save the results
np.save(path_results + 'nulls_' + cancer_name + '_10k_surf_brainspace.npy', rand_data)

#------------------------------------------------------------------------------
# END
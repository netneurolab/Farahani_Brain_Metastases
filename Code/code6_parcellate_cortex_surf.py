"""
###############################################################################

Parcellate cortex data

###############################################################################
"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import scipy.io
import numpy as np
import nibabel as nib
from pathlib import Path
from functions import convert_cifti_to_parcellated_SchaeferTian
from functions import save_parcellated_data_in_SchaeferTian_forVis
from globals import path_medialwall, path_results

#------------------------------------------------------------------------------
# Create paths
#------------------------------------------------------------------------------

cancer_name = 'Melanoma'
mask_medial_wall = scipy.io.loadmat(path_medialwall + 'fs_LR_32k_medial_mask.mat')['medial_mask']
mask_medial_wall = mask_medial_wall.astype(np.float32)

def load_gifti_data(fp: Path) -> np.ndarray:
    """Load a GIFTI file as a 1D float array of vertices."""
    img = nib.load(str(fp))
    data = img.agg_data()
    data = np.asarray(data).squeeze()
    if data.ndim != 1:
        data = np.asarray(data)[:, 0]
    return data.astype(float)

OUT_DIR = "./fs_surface_outputs/";
lh_data = "lh." + cancer_name + "_corrected.32k_fsLR.func.gii"
rh_data = "rh." + cancer_name + "_corrected.32k_fsLR.func.gii"

rh_img = load_gifti_data(OUT_DIR +rh_data)
lh_img = load_gifti_data(OUT_DIR +lh_data)
data_64k = np.concatenate((lh_img, rh_img), axis = 0)
data = data_64k[mask_medial_wall.flatten() == 1]

# Parcellation schaefer-400 for the cortical tissue
data = data.reshape(1,59412)
data_parc = convert_cifti_to_parcellated_SchaeferTian(data, 'cortex', 'an', path_results, cancer_name + '_400')
save_parcellated_data_in_SchaeferTian_forVis(data_parc.flatten(), 'cortex', 'S1', path_results, cancer_name + '_400')

np.save(path_results + cancer_name + '_400_surf.npy', data_parc)

#------------------------------------------------------------------------------
# END
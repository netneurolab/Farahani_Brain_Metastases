"""
###############################################################################

This is the code to pacellate the tumor data in cerebellum and plot it on the flat patch for cerebellum!

    Melanoma: on left hemisphere mostly
    Breast: symmetric - Crus 1 - vermis crus 2
    Lung: symmetric - right VI - Crus 1/2 - vermis

###############################################################################
"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import SUITPy.flatmap as flatmap
from globals import path_MNI09a, path_diedrichsen, path_results, path_nifti

#------------------------------------------------------------------------------
# Settings and helper functions
#------------------------------------------------------------------------------

mni09a = path_MNI09a + 'mni_icbm152_t1_tal_nlin_sym_09a.nii'

atlas_dir = path_results + 'out_reg_ceb/'
cerebellum_atlas_file_name = 'atl-Anatom_space-MNI09aSym_full_dseg.nii.gz'
label_file = path_diedrichsen + "atl-Anatom_dseg.label.gii"

def load_nifti(atlas_path):
    """
    Load nifti data
    """
    out = nib.load(atlas_path)
    return out, out.get_fdata()

#------------------------------------------------------------------------------
# Parcellate the data in cerebellum
#------------------------------------------------------------------------------

atlas_img, atlas_cerebellum = load_nifti(atlas_dir + cerebellum_atlas_file_name) # 197 by 233 by 189
num_labels = len(np.unique(atlas_cerebellum))

for i in range(34):
    print(str(i))
    print(sum(sum(sum(atlas_cerebellum == i + 1))))
    print('------')

# Correct all BM maps
BM_file_names = ['Melanoma_corrected.nii',
                 'Breast_corrected.nii',
                 'Lung_corrected.nii']

parcels = (np.unique(atlas_cerebellum))[1:]

BM_corrected_list = []
for i, bm_file in enumerate(BM_file_names):
    parcelated_map = np.zeros(len(parcels))
    print(f"Correcting {bm_file}...")
    BM_img, BM_data = load_nifti(path_nifti + bm_file) # 197 by 233 by 189
    BM_name = bm_file[:-14]
    for j_index,j in enumerate(parcels):
        parcelated_map[j_index] = np.mean(BM_data[atlas_cerebellum == j])
    np.save(path_results + BM_name + '_34_cerebellum.npy', parcelated_map)

    # Show on the flat patch
    label_data = nib.load(label_file).darrays[0].data
    data_map = np.zeros_like(label_data, dtype=float)
    for i, val in enumerate(parcelated_map):
        mask = (label_data == (i + 1))
        data_map[mask] = val
    flatmap.plot(data=data_map, cmap='coolwarm', colorbar=True, cscale=[0, 0.8*np.max(data_map)])
    plt.savefig(os.path.join(path_results, 'cerebellum_' + BM_name + '.png'), dpi=300)

#------------------------------------------------------------------------------
# END
"""
###############################################################################

    This is the code to pacellate the tumor data in cerebellum and plot it on the falt patch for cerebellum

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
from globals import path_results

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
# Load cerbellum and schaefer atlas
#------------------------------------------------------------------------------

img_schaefer, atlas_schaefer = load_nifti(path_results + 'out_reg_schaefer/schaefer_400-MNI09aSym.nii.gz')

'''
array([[   1.,    0.,    0.,  -98.],
       [   0.,    1.,    0., -134.],
       [   0.,    0.,    1.,  -72.],
       [   0.,    0.,    0.,    1.]])
'''

img_cereb, atlas_cerebellum = load_nifti(path_results + 'out_reg_ceb/atl-Anatom_space-MNI09aSym_full_dseg.nii.gz')
'''
array([[   1.,    0.,    0.,  -98.],
       [   0.,    1.,    0., -134.],
       [   0.,    0.,    1.,  -72.],
       [   0.,    0.,    0.,    1.]])
'''
#------------------------------------------------------------------------------
# Create a template
#------------------------------------------------------------------------------

template = np.zeros(img_cereb.shape, dtype=np.int32)

# cerebellum (1..34)
for j in range(34):
    template[atlas_cerebellum == (j + 1)] = j + 1

# cortex (35..434)
for i in range(400):
    template[atlas_schaefer == (i + 1)] = 35 + i

#------------------------------------------------------------------------------
# Save the combined map
#------------------------------------------------------------------------------

combined_img = nib.Nifti1Image(template, img_cereb.affine, img_cereb.header)
'''
array([[   1.,    0.,    0.,  -98.],
       [   0.,    1.,    0., -134.],
       [   0.,    0.,    1.,  -72.],
       [   0.,    0.,    0.,    1.]])
'''
out_path = os.path.join(path_results, 'cerebellum_cortex_atlas_09.nii.gz')
nib.save(combined_img, out_path)

#------------------------------------------------------------------------------
# END
"""
###############################################################################

This is a code to create a cerebellum atlas in the same dimention and affine.
coordinate as that of MNI09c - sym space.

The result is saved as a file names: "atl-Anatom_space-MNI09c_full_dseg.nii.gz"

###############################################################################
"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import nibabel as nib
from nibabel.processing import resample_from_to
from globals import path_MNI09c, path_diedrichsen

#------------------------------------------------------------------------------
# Helper functions
#------------------------------------------------------------------------------

def load_nifti(atlas_path):
    """
    Load nifti data
    """
    return nib.load(atlas_path).get_fdata()

def resample_to_target(source_img, target_img):
    """
    Resample source image to match target image space
    """
    return resample_from_to(source_img, target_img, order=0)

#------------------------------------------------------------------------------
# Load cerebellum atlas + templates for it
#------------------------------------------------------------------------------

img_mni09c = nib.load(path_MNI09c + 'mni_icbm152_t1_tal_nlin_sym_09c.nii')
data_mni09c = img_mni09c.get_fdata() # the dimentions are 193 by 229 by 193
affine_MNI09c = img_mni09c.affine

# Print the affine matrix
print(f"Affine of MNI09c template:\n{affine_MNI09c}")
'''
Affine of MNI09c template:
[[   1.    0.    0.  -96.]
 [   0.    1.    0. -132.]
 [   0.    0.    1.  -78.]
 [   0.    0.    0.    1.]]
'''

img_ceb = nib.load(path_diedrichsen + 'atl-Anatom_space-MNISym_dseg.nii')
data_ceb = img_ceb.get_fdata() # the dimentions are 153 by 100 by 79
affine_ceb = img_ceb.affine

# Print the affine matrix
print(f"Affine of Ceb atlas:\n{affine_ceb}")
'''
Affine of Ceb atlas:
[[   1.    0.    0.  -76.]
 [   0.    1.    0. -103.]
 [   0.    0.    1.  -70.]
 [   0.    0.    0.    1.]]
'''
#------------------------------------------------------------------------------
# Resample cerebellum atlas to MNI09c space
#------------------------------------------------------------------------------

img_ceb_resampled = resample_to_target(img_ceb, img_mni09c) # the dimentions are 193 by 229 by 193

output_path = path_diedrichsen + 'atl-Anatom_space-MNI09cSym_full_dseg.nii.gz'
nib.save(img_ceb_resampled, output_path)

img_ceb_corr = nib.load(output_path)
data_ceb_corr = img_ceb_corr.get_fdata() # the dimentions are (193, 229, 193)

#------------------------------------------------------------------------------
# END
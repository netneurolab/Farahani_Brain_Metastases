"""
###############################################################################

Load the NIFTI files to varify the template used
    I think the Group-average data is in the same space as the MNI-nLin-Sym-09a
    They have the same dimentions and also they look like each other when opened on workbench
    Note: aSym version also has the same dimentions but when opened in workbench, they do not overlap!

Note:
    Both the T1w and BM data share a -1 in the (0, 0) position, meaning they have a reflection across the X-axis.
    This suggests that both the T1w and BM data are "flipped" in the X direction relative to the MNI template.

Original data - before correction:

    Affine of T1w:
    [[  -1.   -0.    0.   98.]
     [   0.    1.    0. -134.]
     [   0.    0.    1.  -72.]
     [   0.    0.    0.    1.]]

    Affine of MNI template:
    [[   1.    0.    0.  -98.]
     [   0.    1.    0. -134.]
     [   0.    0.    1.  -72.]
     [   0.    0.    0.    1.]]

After correction:

    T1w corrected affine:
    [[   1.    0.    0.  -98.]
     [   0.    1.    0. -134.]
     [   0.    0.    1.  -72.]
     [   0.    0.    0.    1.]]

    BM corrected affine:
    [[   1.    0.    0.  -98.]
     [   0.    1.    0. -134.]
     [   0.    0.    1.  -72.]
     [   0.    0.    0.    1.]]

    MNI template affine:
    [[   1.    0.    0.  -98.]
     [   0.    1.    0. -134.]
     [   0.    0.    1.  -72.]
     [   0.    0.    0.    1.]]

###############################################################################
"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import numpy as np
import nibabel as nib
from globals import path_nifti, path_MNI09a

#------------------------------------------------------------------------------
# Helper functions
#------------------------------------------------------------------------------

def load_nifti(atlas_path):
    """
    Load nifti data
    """
    return nib.load(atlas_path).get_fdata()

def flip_and_correct_affine(nifti_img, target_affine):
    """
    Flip data along X-axis and correct affine to match target

    Parameters:
    -----------
    nifti_img : nibabel image object
        The image to correct
    target_affine : array
        The target affine matrix (e.g., from MNI template)

    Returns:
    --------
    corrected_img : nibabel image object
        Image with flipped data and corrected affine
    """
    # Get the data
    data = nifti_img.get_fdata()

    # Check if X-axis needs flipping
    if np.sign(nifti_img.affine[0, 0]) != np.sign(target_affine[0, 0]):
        print("Flipping data along X-axis.")
        data_flipped = np.flip(data, axis = 0)
    else:
        print("No flip needed.")
        data_flipped = data

    # Create new image with corrected affine
    corrected_img = nib.Nifti1Image(data_flipped, target_affine, nifti_img.header)

    return corrected_img

#------------------------------------------------------------------------------
# what is the space where BM data is provided? MNI2009Sym-a - but affines are different
#------------------------------------------------------------------------------

# Load the T1w data provided in BM OSF folder
img = nib.load(path_nifti + 'T1w_MNI.nii')
data_img = img.get_fdata() # the dimentions are 197 by 233 by 189

# Open the MNI template - this will be our reference
img_MNI = nib.load(path_MNI09a + 'mni_icbm152_t1_tal_nlin_sym_09a.nii')
data_MNI = img_MNI.get_fdata() # the dimentions are 197 by 233 by 189

# Get the affine matrices
affine_T1w = img.affine
affine_MNI = img_MNI.affine

# Print out the affine matrices to inspect
print("BEFORE CORRECTION:")
print(f"Affine of T1w:\n{affine_T1w}")
print(f"Affine of MNI template:\n{affine_MNI}")

#------------------------------------------------------------------------------
# Correct them to have all with the same affine (use MNI as reference)
#------------------------------------------------------------------------------

print("\n" + "="*80)
print("CORRECTING AFFINE MATRICES TO MATCH MNI TEMPLATE")
print("="*80 + "\n")

# Correct T1w
img_T1w_corrected = flip_and_correct_affine(img, affine_MNI)
nib.save(img_T1w_corrected, path_nifti + 'T1w_MNI_corrected.nii')

# Correct all BM maps
BM_file_names = ['Melanoma.nii',
                 'Breast.nii',
                 'Lung.nii'] # these have the same affine as the T1w data - so they also need to be converted

num_maps = len(BM_file_names)

BM_corrected_list = []
for i, bm_file in enumerate(BM_file_names):
    print(f"Correcting {bm_file}...")
    BM_img = nib.load(path_nifti + bm_file)

    # Print original affine
    if i == 0:  # only print once for first file
        print(f"Original BM affine:\n{BM_img.affine}")

    # Correct the image
    BM_corrected = flip_and_correct_affine(BM_img, affine_MNI)
    BM_corrected_list.append(BM_corrected)

    # Save corrected version
    corrected_filename = bm_file.replace('.nii', '_corrected.nii')
    nib.save(BM_corrected, path_nifti + corrected_filename)
    print(f"Saved: {corrected_filename}\n")

#------------------------------------------------------------------------------
# Verify corrections
#------------------------------------------------------------------------------

print("="*80)
print("VERIFICATION")
print("="*80 + "\n")

# Load one corrected file to verify
BM_verified = nib.load(path_nifti + 'Melanoma_corrected.nii')
T1w_verified = nib.load(path_nifti + 'T1w_MNI_corrected.nii')
MNI_verified = nib.load(path_MNI09a + 'mni_icbm152_t1_tal_nlin_sym_09a.nii')
print("After correction:")
print(f"T1w corrected affine:\n{T1w_verified.affine}")
print(f"\nBM corrected affine:\n{BM_verified.affine}")
print(f"\nMNI template affine:\n{MNI_verified.affine}")

#------------------------------------------------------------------------------
# END
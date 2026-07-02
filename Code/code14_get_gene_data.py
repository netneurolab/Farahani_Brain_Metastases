"""
###############################################################################

Get the gene maps from abagen toolbox

###############################################################################
"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import abagen
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.io import savemat
from nilearn.image import resample_to_img
from globals import path_results, path_data

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
# Load cerebellum atlas
#------------------------------------------------------------------------------

atlas_dir = path_results + 'out_reg_hcp_MNI06_ceb/'
cerebellum_atlas_file_name = 'atl-Anatom_space-MNI06_full_dseg.nii.gz'
img_ceb, atlas_cerebellum = load_nifti(atlas_dir + cerebellum_atlas_file_name) # 182 by 218 by 182

img_ceb = nib.load(os.path.join(atlas_dir, 'atl-Anatom_space-MNI06_full_dseg.nii.gz'))
img_sch = nib.load(path_data + 'Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz')
img_sch_on_ceb = resample_to_img(img_sch, img_ceb, interpolation='nearest')

sch = img_sch_on_ceb.get_fdata()

#------------------------------------------------------------------------------
# Create a template
#------------------------------------------------------------------------------

template = np.zeros(img_ceb.shape, dtype=np.int32)

# cerebellum
for j in range(32):
    template[atlas_cerebellum == (j + 1)] = j + 1

# cortex (35..434)
for i in range(1, 401):
    template[sch == i] = 32 + i

combined_img = nib.Nifti1Image(template, img_ceb.affine, img_ceb.header)

out_path = os.path.join(path_results, 'cerebellum32_cortex_atlas_06.nii.gz')
nib.save(combined_img, out_path)

#------------------------------------------------------------------------------
# Get the gene maps
#------------------------------------------------------------------------------

atlas_dir = path_results + 'cerebellum32_cortex_atlas_06.nii.gz'

expression = abagen.get_expression_data(atlas_dir,
                                        lr_mirror     = 'bidirectional',
                                        missing       = 'interpolate',
                                        return_donors = True)

expression_st, ds = abagen.correct.keep_stable_genes(list(expression.values()),
                                                  threshold        = 0.1,
                                                  percentile       = False,
                                                  return_stability = True)

expression_st = pd.concat(expression_st).groupby('label').mean()

#------------------------------------------------------------------------------
# Save the obtained values
#------------------------------------------------------------------------------

columns_name = np.array(expression_st.columns)
data_to_save = {'names': columns_name}
savemat(path_results + 'names_genes.mat', data_to_save)
np.save(path_results + 'names_genes.npy', columns_name)

data_to_save = {'gene_coexpression': np.array(expression_st)}
savemat(path_results + 'gene_coexpression.mat', data_to_save)
np.save(path_results + 'gene_coexpression.npy', np.array(expression_st))

#------------------------------------------------------------------------------
# END
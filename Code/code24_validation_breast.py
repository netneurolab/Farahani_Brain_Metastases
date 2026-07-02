"""
###############################################################################
Code of the verification:

    Breast cancer data from another sample

    spearman r = 0.4770221545873529
    p = 0.001998001998001998

###############################################################################
"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import scipy.io
import numpy as np
import nibabel as nib
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import SUITPy.flatmap as flatmap
from scipy.stats import spearmanr
from functions import convert_cifti_to_parcellated_SchaeferTian
from functions import save_parcellated_data_in_SchaeferTian_forVis, pval_cal
from globals import path_diedrichsen, path_results, path_validation, path_medialwall

#------------------------------------------------------------------------------
# What is the space where BM data is provided? MNI2009Sym-a - but affines are different
#------------------------------------------------------------------------------

mask_medial_wall = scipy.io.loadmat(path_medialwall + 'fs_LR_32k_medial_mask.mat')['medial_mask']
mask_medial_wall = mask_medial_wall.astype(np.float32)

def load_gifti_data(fp: Path) -> np.ndarray:
    """Load a GIFTI as a 1D float array of vertices."""
    img = nib.load(str(fp))
    data = img.agg_data()
    data = np.asarray(data).squeeze()
    if data.ndim != 1:
        data = np.asarray(data)[:, 0]
    return data.astype(float)

cancer_name = 'Breast_distribution_HEAD-MNI09aSym.nii'
cancer_name_short = 'Breast_distribution_HEAD-MNI09aSym_surf'

OUT_DIR = "./fs_surface_outputs/";
lh_data = "lh." + cancer_name + ".32k_fsLR.func.gii"
rh_data = "rh." + cancer_name + ".32k_fsLR.func.gii"

rh_img = load_gifti_data(OUT_DIR +rh_data)
lh_img = load_gifti_data(OUT_DIR +lh_data)
data_64k = np.concatenate((lh_img, rh_img), axis =0)
data = data_64k[mask_medial_wall.flatten() == 1]

# Parcellation schaefer-400 for cortical tissue
data = data.reshape(1, 59412)
BM_parcellated = convert_cifti_to_parcellated_SchaeferTian(data,
                                                           'cortex', 'an', path_results, cancer_name_short + '_400')
save_parcellated_data_in_SchaeferTian_forVis(BM_parcellated.flatten(),
                                             'cortex', 'S1', path_results, cancer_name_short + '_400')
np.save(path_results + cancer_name_short + '_400.npy', BM_parcellated)

#------------------------------------------------------------------------------
# Correct them to have all with the same affine (use MNI as reference)
#------------------------------------------------------------------------------

print("\n" + "="*80)
print("CORRECTING AFFINE MATRICES TO MATCH MNI TEMPLATE")
print("="*80 + "\n")

BM_img = nib.load(path_validation + 'Breast_distribution_HEAD.nii')

# Print original affine
print(f" Original BM affine:\n{BM_img.affine}")

#------------------------------------------------------------------------------
# Parcellate data - cortical and cerebellum
#------------------------------------------------------------------------------

# Cerebellum
output_path = path_diedrichsen + 'atl-Anatom_space-MNI09cSym_full_dseg.nii.gz'
cereb_atlas = nib.load(output_path)
print(f"\nMNI template affine:\n{cereb_atlas.affine}")

BM_parcellated_cereb = np.zeros(34)
for i in range(34):
    BM_parcellated_cereb[i] = np.nanmean(BM_img.get_fdata()[cereb_atlas.get_fdata()==i+1])

#------------------------------------------------------------------------------
# Visualize parcellated data on cerebellum
#------------------------------------------------------------------------------

atlas_dir = path_results + 'out_reg_ceb/'
cerebellum_atlas_file_name = 'atl-Anatom_space-MNI09aSym_full_dseg.nii.gz'
label_file = path_diedrichsen + "atl-Anatom_dseg.label.gii"

def load_nifti(atlas_path):
    """
    Load nifti data
    """
    out = nib.load(atlas_path)
    return out, out.get_fdata()

atlas_img, atlas_cerebellum = load_nifti(atlas_dir + cerebellum_atlas_file_name) # 197 by 233 by 189
num_labels = len(np.unique(atlas_cerebellum))

label_data = nib.load(label_file).darrays[0].data

data_map = np.zeros_like(label_data, dtype=float)
for i, val in enumerate(BM_parcellated_cereb):
    mask = (label_data == (i + 1))
    data_map[mask] = val

flatmap.plot(data=data_map, cmap='coolwarm', colorbar=True)
verification_data = np.concatenate((BM_parcellated_cereb[:32], BM_parcellated.flatten()), axis = 0)
np.save(path_results + 'replication_Breast_HEAD.npy', verification_data)

#------------------------------------------------------------------------------
# Load main data
#------------------------------------------------------------------------------

main_nulls = np.load(path_results + 'nulls_Breast_10k_surf_brainspace.npy')[:1000,:]
main_cancer_parc_cereb = np.load(path_results + 'Breast_34_cerebellum.npy')
main_cancer_parc_cortex = np.load(path_results + 'Breast_400_surf.npy').flatten()
main_data = np.concatenate((main_cancer_parc_cereb[:32], main_cancer_parc_cortex), axis = 0)

#------------------------------------------------------------------------------
# How similar are the methasthasis maps of the two groups?
#------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 7))
sns.regplot(x = main_data, y = verification_data,
            color='silver',
            scatter_kws={'s': 80, 'edgecolor': 'black', 'linewidth': 0.8})
plt.scatter(main_data[:32], verification_data[:32], color = 'black', s  = 80)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(path_results + "validation_Breast.svg", format="svg")
plt.show()

print(spearmanr(main_data, verification_data))

# Assess significance of the colocalization
nspins = 1000
r = spearmanr(main_data, verification_data)[0]
corr_null = np.zeros(nspins)
for i in range(nspins):
    corr_null[i] = spearmanr(main_nulls[i,:], verification_data)[0]
print(pval_cal(r, corr_null, nspins))

#------------------------------------------------------------------------------
# END